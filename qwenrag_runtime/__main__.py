"""Enable ``python -m qwenrag_runtime`` during source development."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
