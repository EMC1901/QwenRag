"""Immutable Base + Delta manifest parsing and atomic publication."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .persistence import atomic_write_text


class ManifestError(RuntimeError):
    """Raised when a knowledge manifest is malformed or changed concurrently."""


@dataclass(frozen=True)
class LayerSpec:
    layer_id: str
    relative_path: str
    meta_sha256: str | None = None
    published_at: str | None = None


@dataclass(frozen=True)
class KnowledgeManifest:
    schema_version: int
    revision: int
    base: LayerSpec
    deltas: tuple[LayerSpec, ...]
    next_vector_id: int
    embedding_model: str
    embedding_revision: str
    embedding_dim: int
    vector_metric: str
    vector_normalized: bool


def load_manifest(root: Path, manifest_path: Path | None = None) -> KnowledgeManifest | None:
    """Read and validate the currently published manifest, if any."""

    path = manifest_path or root / "knowledge_manifest.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError("MANIFEST_INVALID") from exc
    if not isinstance(raw, dict):
        raise ManifestError("MANIFEST_INVALID")
    try:
        base = _parse_layer(raw["base"], "generation_id")
        deltas_raw = raw.get("deltas")
        if not isinstance(deltas_raw, list):
            raise ValueError
        deltas = tuple(_parse_layer(value, "delta_id") for value in deltas_raw)
        result = KnowledgeManifest(
            schema_version=_positive_int(raw["schema_version"]),
            revision=_nonnegative_int(raw["revision"]),
            base=base,
            deltas=deltas,
            next_vector_id=_nonnegative_int(raw["next_vector_id"]),
            embedding_model=_nonempty_text(raw["embedding_model"]),
            # Older development knowledge bases did not record a model
            # revision.  Preserve their readability while every delivered
            # Base/Deltas records the exact installed revision.
            embedding_revision=_nonempty_text(raw.get("embedding_revision", "legacy-unknown")),
            embedding_dim=_positive_int(raw["embedding_dim"]),
            vector_metric=_nonempty_text(raw["vector_metric"]),
            vector_normalized=_bool(raw["vector_normalized"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError("MANIFEST_INVALID") from exc
    _validate_manifest_paths(root, result)
    if len({layer.layer_id for layer in result.deltas}) != len(result.deltas):
        raise ManifestError("MANIFEST_DUPLICATE_DELTA")
    return result


def legacy_base_manifest(root: Path, *, embedding_model: str, embedding_dim: int, next_vector_id: int) -> KnowledgeManifest:
    """Represent the existing root layout as an implicit immutable Base."""

    return KnowledgeManifest(
        schema_version=1,
        revision=0,
        base=LayerSpec("legacy_root", "."),
        deltas=(),
        next_vector_id=next_vector_id,
        embedding_model=embedding_model,
        embedding_revision="legacy-unknown",
        embedding_dim=embedding_dim,
        vector_metric="inner_product",
        vector_normalized=True,
    )


def publish_delta(
    root: Path,
    manifest_path: Path,
    delta: LayerSpec,
    *,
    expected_revision: int,
    embedding_model: str,
    embedding_dim: int,
    next_vector_id: int,
) -> KnowledgeManifest:
    """Append one fully validated Delta with a compare-and-swap manifest update."""

    current = load_manifest(root, manifest_path)
    if current is None:
        current = legacy_base_manifest(
            root,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            next_vector_id=0,
        )
    if current.revision != expected_revision:
        raise ManifestError("MANIFEST_CONFLICT")
    if current.embedding_model != embedding_model or current.embedding_dim != embedding_dim:
        raise ManifestError("MANIFEST_EMBEDDING_MISMATCH")
    if any(item.layer_id == delta.layer_id for item in current.deltas):
        raise ManifestError("MANIFEST_DELTA_ALREADY_PRESENT")
    result = KnowledgeManifest(
        schema_version=1,
        revision=current.revision + 1,
        base=current.base,
        deltas=(*current.deltas, delta),
        next_vector_id=next_vector_id,
        embedding_model=current.embedding_model,
        embedding_revision=current.embedding_revision,
        embedding_dim=current.embedding_dim,
        vector_metric=current.vector_metric,
        vector_normalized=current.vector_normalized,
    )
    _validate_manifest_paths(root, result)
    atomic_write_text(manifest_path, json.dumps(_to_json(result), ensure_ascii=False, indent=2) + "\n")
    return result


def layer_directory(root: Path, layer: LayerSpec) -> Path:
    """Resolve an already validated relative layer directory inside ``root``."""

    candidate = (root / layer.relative_path).resolve(strict=True)
    candidate.relative_to(root.resolve(strict=True))
    return candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_layer(raw: Any, id_key: str) -> LayerSpec:
    if not isinstance(raw, dict):
        raise ValueError
    layer_id = _nonempty_text(raw[id_key])
    relative_path = _nonempty_text(raw["relative_path"])
    meta_sha256 = raw.get("meta_sha256")
    if meta_sha256 is not None and (not isinstance(meta_sha256, str) or len(meta_sha256) != 64):
        raise ValueError
    published_at = raw.get("published_at")
    if published_at is not None and not isinstance(published_at, str):
        raise ValueError
    return LayerSpec(layer_id, relative_path, meta_sha256, published_at)


def _validate_manifest_paths(root: Path, manifest: KnowledgeManifest) -> None:
    for layer in (manifest.base, *manifest.deltas):
        relative = Path(layer.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ManifestError("MANIFEST_PATH_INVALID")
        resolved = (root / relative).resolve(strict=False)
        try:
            resolved.relative_to(root.resolve(strict=False))
        except ValueError as exc:
            raise ManifestError("MANIFEST_PATH_INVALID") from exc


def _to_json(manifest: KnowledgeManifest) -> dict[str, object]:
    def layer_json(layer: LayerSpec, id_key: str) -> dict[str, object]:
        result: dict[str, object] = {id_key: layer.layer_id, "relative_path": layer.relative_path}
        if layer.meta_sha256 is not None:
            result["meta_sha256"] = layer.meta_sha256
        if layer.published_at is not None:
            result["published_at"] = layer.published_at
        return result

    return {
        "schema_version": manifest.schema_version,
        "revision": manifest.revision,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "base": layer_json(manifest.base, "generation_id"),
        "deltas": [layer_json(layer, "delta_id") for layer in manifest.deltas],
        "next_vector_id": manifest.next_vector_id,
        "embedding_model": manifest.embedding_model,
        "embedding_revision": manifest.embedding_revision,
        "embedding_dim": manifest.embedding_dim,
        "vector_metric": manifest.vector_metric,
        "vector_normalized": manifest.vector_normalized,
    }


def _positive_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError
    return value


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError
    return value


def _nonempty_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError
    return value


def _bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError
    return value
