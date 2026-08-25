"""Small offline entry point used by the PowerShell release pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


# This script is executed by an absolute path from the packaging directory.
# Make the repository importable without asking release engineers to set
# PYTHONPATH manually.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qwenrag_runtime.kb_snapshot import create_kb_snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--embedding-revision")
    args = parser.parse_args()
    result = create_kb_snapshot(
        args.source,
        args.destination,
        version=args.version,
        embedding_revision=args.embedding_revision,
    )
    print(json.dumps({"snapshot": str(result)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
