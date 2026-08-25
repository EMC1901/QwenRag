"""Incremental task orchestration through Base + Delta publication."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import shutil
from pathlib import Path
from rag_preprocess.processing import process_document, write_intermediate
from .intake import classify, required_free_bytes, scan_and_freeze, service_stop_issues, verify_ready
from .persistence import write_checkpoint, write_status
from .embedding import EmbeddingPreflightError, embed_file_chunks, preflight_embedding
from .candidate import archive_files
from .delta import build_task_delta, load_effective_identities
from .delta_index import build_delta_indexes, publish_delta_package, validate_delta_package
from .manifest import layer_directory, load_manifest
from .reporting import write_final_result
from .parsers.base import ParseFailure
from .parsers.docx_adapter import parse_docx_v2
from .parsers.pdf_ocr import OfflinePaddleOcr
from .parsers.pdf_parser import parse_pdf
from .parsers.txt_parser import parse_txt_v2

def _write_task_state(task_path: Path, task_id: str, state: str, **details: object) -> None:
    write_checkpoint(task_path, {"schema_version": 1, "task_id": task_id, "state": state, **details})

def run_stages_4_to_9(settings, task_id: str) -> dict[str, object]:
    """Freeze and parse a batch, retaining only versioned internal artefacts.

    Publishing/archiving remains deliberately outside this stage range.
    """
    work=settings.work_dir/task_id; task_path=work/'task.json'; manifest=work/'files.json'
    started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    manifest_state = load_manifest(settings.knowledge_base_root, settings.manifest_path)
    base_root = settings.knowledge_base_root if manifest_state is None else layer_directory(settings.knowledge_base_root, manifest_state.base)
    base_database = base_root / 'metadata.db'
    delta_databases = [] if manifest_state is None else [layer_directory(settings.knowledge_base_root, layer) / 'delta.db' for layer in manifest_state.deltas]
    assets=[base_database,base_root/'vector_index'/'index.faiss',base_root/'vector_index'/'index.meta.json',*delta_databases]
    _write_task_state(task_path, task_id, "PREFLIGHT")
    issues=service_stop_issues(settings.local_rag_host,settings.local_rag_port,settings.health_url,assets)
    if issues:
        payload={"schema_version":1,"task_id":task_id,"state":"REJECTED_SERVICE_RUNNING","issues":[issue.code for issue in issues]}; write_checkpoint(task_path,payload); write_status(settings.results_dir/f'{task_id}.status.txt','拒绝执行：本地检索服务仍在运行。'); return payload
    required=required_free_bytes([],sum(p.stat().st_size for p in settings.incoming_dir.iterdir() if p.is_file()))
    if shutil.disk_usage(settings.knowledge_base_root).free < required:
        payload={"schema_version":1,"task_id":task_id,"state":"FAILED_RESUMABLE","error_code":"DISK_SPACE_LOW"}; write_checkpoint(task_path,payload); return payload
    _write_task_state(task_path, task_id, "SNAPSHOTTING")
    rows=scan_and_freeze(settings.incoming_dir,manifest)
    for row in rows:
        if row.state=='FROZEN': verify_ready(row,probes=settings.file_stability_probe_count,interval_seconds=settings.file_stability_probe_interval_seconds)
    identities=load_effective_identities(base_database,delta_databases=delta_databases) if base_database.exists() else {}
    classify(rows,identities); from .intake import write_frozen_manifest; write_frozen_manifest(manifest,rows)
    processable=[row for row in rows if row.action in {'NEW','UPDATE'}]
    if processable:
        try:
            _write_task_state(task_path, task_id, "PREFLIGHT", current_operation="embedding")
            preflight_embedding(settings)
        except EmbeddingPreflightError as exc:
            payload={"schema_version":1,"task_id":task_id,"state":"FAILED_RESUMABLE","error_code":str(exc)}
            write_checkpoint(task_path,payload)
            write_status(settings.results_dir/f'{task_id}.status.txt','Embedding 服务预检失败；未修改 Base、Delta 或正式知识库。')
            return payload
    completed=0; failed=0; finished=0; publication=None
    _write_task_state(task_path, task_id, "PROCESSING_FILES", total_file_count=len(processable), processed_file_count=0, successful_file_count=0, failed_file_count=0)
    for row in rows:
        if row.action not in {'NEW','UPDATE'}: continue
        _write_task_state(task_path, task_id, "PROCESSING_FILES", total_file_count=len(processable), processed_file_count=finished, successful_file_count=completed, failed_file_count=failed, current_file_name=row.file_name)
        try:
            path=Path(row.frozen_path)
            if row.extension=='.docx': document=parse_docx_v2(path,row.doc_id or '')
            elif row.extension=='.txt': document=parse_txt_v2(path,row.doc_id or '')
            else:
                ocr=OfflinePaddleOcr(settings.ocr_model_dir,settings.ocr_text_detection_model,settings.ocr_text_recognition_model,settings.ocr_cpu_threads)
                document=parse_pdf(path,row.doc_id or '',ocr=ocr,work_dir=work,checkpoint_path=work/f'{row.version_id}.pdf.checkpoint.json')
            row.title=document.title
            write_intermediate(work/f'{row.version_id}.parsed.json',document)
            result=process_document(document)
            write_checkpoint(work/f'{row.version_id}.chunks.json',{'schema_version':1,'chunk_count':len(result.chunks),'rejected_blocks':result.rejected_blocks,'chunks':[asdict(chunk) for chunk in result.chunks]})
            row.state='EMBEDDING'
            embed_file_chunks(settings, row.version_id or '', result.chunks, work/'vectors')
            row.state='READY_TO_DELTA'; completed+=1
        except ParseFailure as exc:
            row.state='FAILED'; row.error_code=exc.code; failed+=1
        except Exception:
            # A local parser/model failure must not strand an entire customer
            # batch or hide all prior per-file results.
            row.state='FAILED'; row.error_code='PARSER_RUNTIME_FAILED'; failed+=1
        finally:
            finished+=1
            write_frozen_manifest(manifest,rows)
            _write_task_state(task_path, task_id, "PROCESSING_FILES", total_file_count=len(processable), processed_file_count=finished, successful_file_count=completed, failed_file_count=failed)
    write_frozen_manifest(manifest,rows)
    if completed:
        try:
            _write_task_state(task_path, task_id, 'BUILDING_DELTA_DB', successful_file_count=completed, failed_file_count=failed)
            ready=[row.__dict__ for row in rows if row.state=='READY_TO_DELTA']
            delta_root, _delta_results, _delta_metadata = build_task_delta(
                settings.work_dir, task_id, ready, base_database=base_database,
                prior_delta_databases=delta_databases, embedding_dim=settings.embedding_dim,
                embedding_model=settings.embedding_model,
                first_vector_id=manifest_state.next_vector_id if manifest_state is not None else None,
                parent_manifest_revision=manifest_state.revision if manifest_state is not None else None,
            )
            for row in rows:
                if row.state=='READY_TO_DELTA': row.state='DELTA_BUILT'
            _write_task_state(task_path,task_id,'BUILDING_DELTA_FTS')
            build_delta_indexes(delta_root,embedding_dim=settings.embedding_dim,embedding_model=settings.embedding_model,embedding_revision=settings.embedding_revision)
            _write_task_state(task_path,task_id,'BUILDING_DELTA_FAISS')
            _write_task_state(task_path,task_id,'VALIDATING_DELTA')
            validate_delta_package(delta_root,embedding_dim=settings.embedding_dim,embedding_model=settings.embedding_model,embedding_revision=settings.embedding_revision)
            for row in rows:
                if row.state=='DELTA_BUILT': row.state='DELTA_VALIDATED'
            _write_task_state(task_path,task_id,'PUBLISHING')
            publication=publish_delta_package(settings,delta_root,expected_revision=manifest_state.revision if manifest_state is not None else 0)
            for row in rows:
                if row.state=='DELTA_VALIDATED':
                    row.state='PUBLISHED'
                    row.delta_id=publication.deltas[-1].layer_id
                    row.manifest_revision=publication.revision
                    row.published_at=publication.deltas[-1].published_at
        except Exception as exc:
            failed += completed; completed=0
            for row in rows:
                if row.state in {'READY_TO_DELTA','DELTA_BUILT','DELTA_VALIDATED'}: row.state='FAILED'; row.error_code=str(exc)
    archive_failed=0
    archivable=[row for row in rows if row.state in {'PUBLISHED','DUPLICATE_UNCHANGED'}]
    if archivable:
        _write_task_state(task_path,task_id,'ARCHIVING')
        archive_outcomes=archive_files(settings,[row.__dict__ for row in archivable],task_id)
        for row in archivable:
            outcome=archive_outcomes.get(str(row.version_id or row.frozen_path or row.file_name))
            if outcome is not None and outcome.archived:
                row.state='ARCHIVED'; row.archive_relative_path=outcome.relative_path
            else:
                row.error_code=outcome.error_code if outcome is not None else 'ARCHIVE_FAILED'
                if row.state=='PUBLISHED':
                    row.state='PUBLISHED_ARCHIVE_FAILED'; archive_failed+=1
                else:
                    row.state='FAILED'; failed+=1
    write_frozen_manifest(manifest,rows)
    archived_count=sum(row.state=='ARCHIVED' for row in rows)
    if not processable and not failed:
        state='NO_CHANGES'
    elif (completed or archived_count) and not failed and not archive_failed:
        state='SUCCEEDED'
    elif completed or archived_count:
        state='PARTIAL_SUCCESS'
    elif not processable:
        state='NO_CHANGES'
    else:
        state='FAILED_RESUMABLE'
    payload={'schema_version':1,'task_id':task_id,'state':state,'total_file_count':len(processable),'processed_file_count':finished,'successful_file_count':completed,'failed_file_count':failed}; write_checkpoint(task_path,payload); write_status(settings.results_dir/f'{task_id}.status.txt',f'增量入库完成：成功发布 {completed} 个文件，失败 {failed} 个。')
    payload.update({
        'state':state,
        'started_at':started_at,
        'finished_at':datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
        'archived_file_count':archived_count,
        'archive_pending_file_count':archive_failed,
        'failed_file_count':failed,
        'delta_id':publication.deltas[-1].layer_id if publication is not None else None,
        'manifest_revision':publication.revision if publication is not None else None,
        'validation':'passed' if publication is not None else 'not_applicable',
        'embedding_model':settings.embedding_model,
        'embedding_dim':settings.embedding_dim,
    })
    write_checkpoint(task_path,payload)
    write_status(
        settings.results_dir/f'{task_id}.status.txt',
        f"增量入库完成：已发布 {completed} 个文件，已归档 {archived_count} 个文件，待补归档 {archive_failed} 个文件，失败 {failed} 个。",
    )
    write_final_result(settings.results_dir/f'{task_id}.result.txt',task_id,rows,task=payload)
    return payload
