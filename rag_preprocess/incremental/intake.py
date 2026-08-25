"""Stage 4: safe preflight, one-level intake freezing and classification."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib, math, os, socket, sqlite3, time, unicodedata
from pathlib import Path
from typing import Callable, Iterable

from .persistence import write_checkpoint

SUPPORTED_EXTENSIONS = {".docx", ".pdf", ".txt"}

class IntakeAction(str, Enum):
    NEW = "NEW"; UPDATE = "UPDATE"; DUPLICATE_UNCHANGED = "DUPLICATE_UNCHANGED"

@dataclass(frozen=True)
class IntakeIssue:
    code: str; message: str

@dataclass
class FrozenFile:
    file_name: str; logical_name_key: str; frozen_path: str; size: int; mtime_ns: int
    sha256: str | None = None; extension: str = ""; action: str | None = None
    state: str = "FROZEN"; warning_codes: list[str] = field(default_factory=list)
    error_code: str | None = None; doc_id: str | None = None; version_id: str | None = None
    title: str | None = None; archive_relative_path: str | None = None
    delta_id: str | None = None; manifest_revision: int | None = None; published_at: str | None = None

def logical_name_key(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()

def stable_doc_id(key: str) -> str:
    return "doc_" + hashlib.sha256(("incremental:" + key).encode()).hexdigest()[:24]

def version_id(doc_id: str, digest: str) -> str:
    return "ver_" + hashlib.sha256((doc_id + ":" + digest).encode()).hexdigest()[:24]

def is_reparse_point(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path.stat(), "st_file_attributes", 0) & 0x400)

def scan_and_freeze(incoming_dir: Path, manifest_path: Path) -> list[FrozenFile]:
    """Snapshot direct children only. Files arriving afterwards are deliberately absent."""
    records: list[FrozenFile] = []
    candidates: dict[str, list[FrozenFile]] = {}
    for path in incoming_dir.iterdir():
        if path.name.startswith("~$"):
            continue
        if is_reparse_point(path):
            records.append(FrozenFile(path.name, logical_name_key(path.name), str(path), 0, 0, state="REJECTED", error_code="REPARSE_POINT")); continue
        if path.is_dir():
            records.append(FrozenFile(path.name, logical_name_key(path.name), str(path), 0, 0, state="IGNORED", error_code="SUBDIRECTORY")); continue
        stat = path.stat(); ext = path.suffix.lower()
        row = FrozenFile(path.name, logical_name_key(path.name), str(path), stat.st_size, stat.st_mtime_ns, extension=ext,
                         state="UNSUPPORTED" if ext not in SUPPORTED_EXTENSIONS else "FROZEN")
        records.append(row)
        if row.state == "FROZEN": candidates.setdefault(row.logical_name_key, []).append(row)
    for same in candidates.values():
        if len(same) > 1:
            for row in same: row.state, row.error_code = "REJECTED", "NAME_COLLISION"
    write_frozen_manifest(manifest_path, records)
    return records

def _stat_token(path: Path) -> tuple[int, int, int]:
    s = path.stat(); return s.st_size, s.st_mtime_ns, getattr(s, "st_ino", 0)

def exclusive_openable(path: Path) -> bool:
    """Use CreateFileW with no sharing on Windows; ordinary open is insufficient there."""
    if os.name != "nt":
        try:
            with path.open("rb"): return True
        except OSError: return False
    import ctypes
    handle = ctypes.windll.kernel32.CreateFileW(str(path), 0x80000000, 0, None, 3, 0, None)
    if handle == -1: return False
    ctypes.windll.kernel32.CloseHandle(handle); return True

def sha256_stream(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(chunk_size), b""): digest.update(data)
    return digest.hexdigest()

def verify_ready(row: FrozenFile, *, probes: int = 3, interval_seconds: float = 2, sleeper: Callable[[float], None] = time.sleep) -> FrozenFile:
    path = Path(row.frozen_path)
    try:
        tokens = []
        for index in range(probes):
            tokens.append(_stat_token(path))
            if index + 1 < probes: sleeper(interval_seconds)
        if len(set(tokens)) != 1 or not exclusive_openable(path): raise OSError("unstable or busy")
        digest = sha256_stream(path)
        if _stat_token(path) != tokens[-1]: raise OSError("changed while hashing")
        row.size, row.mtime_ns, row.sha256 = tokens[-1][0], tokens[-1][1], digest
    except OSError:
        row.state, row.error_code = "NOT_READY", "FILE_NOT_READY"
    return row

def classify(rows: Iterable[FrozenFile], identities: dict[str, tuple[str, str]], hashes: dict[str, str] | None = None) -> list[FrozenFile]:
    hashes = hashes or {digest: key for key, (_, digest) in identities.items()}
    for row in rows:
        if row.state != "FROZEN" or not row.sha256: continue
        prior = identities.get(row.logical_name_key)
        row.doc_id = prior[0] if prior else stable_doc_id(row.logical_name_key)
        row.action = IntakeAction.DUPLICATE_UNCHANGED if prior and prior[1] == row.sha256 else (IntakeAction.UPDATE if prior else IntakeAction.NEW)
        row.version_id = version_id(row.doc_id, row.sha256)
        other = hashes.get(row.sha256)
        if other and other != row.logical_name_key: row.warning_codes.append("DUPLICATE_CONTENT_DIFFERENT_NAME")
    return list(rows)

def write_frozen_manifest(path: Path, rows: Iterable[FrozenFile]) -> None:
    write_checkpoint(path, {"schema_version": 1, "files": [asdict(row) for row in rows]})

def load_legacy_identities(metadata_db: Path) -> dict[str, tuple[str, str]]:
    """Read legacy basename identities and fail closed on NFC/casefold collisions."""
    with sqlite3.connect(f"file:{metadata_db}?mode=ro", uri=True) as con:
        rows = con.execute("SELECT doc_id, relative_path, file_hash_sha256 FROM documents").fetchall()
    result: dict[str, tuple[str, str]] = {}
    for doc_id, relative_path, digest in rows:
        key = logical_name_key(Path(relative_path).name)
        if key in result: raise ValueError(f"存量文件名冲突：{key}")
        result[key] = (doc_id, digest)
    return result

def required_free_bytes(asset_sizes: Iterable[int], batch_input_bytes: int, journal_bytes: int = 0) -> int:
    return math.ceil((sum(asset_sizes) + batch_input_bytes * 3 + journal_bytes) * 1.2)

def service_stop_issues(host: str, port: int, health_url: str, asset_paths: Iterable[Path], *, http_get=None) -> list[IntakeIssue]:
    issues=[]
    try:
        with socket.create_connection((host, port), timeout=.3): issues.append(IntakeIssue("LOCAL_RAG_PORT_ACTIVE", "本地检索端口仍在监听"))
    except OSError: pass
    if http_get:
        try:
            response=http_get(health_url, timeout=.5)
            if getattr(response, "status_code", 500) < 500: issues.append(IntakeIssue("LOCAL_RAG_HEALTH_ACTIVE", "健康检查仍可访问"))
        except Exception: pass
    for asset in asset_paths:
        if asset.exists() and not exclusive_openable(asset): issues.append(IntakeIssue("KNOWLEDGE_ASSET_LOCKED", f"知识库文件被占用：{asset.name}"))
    return issues
