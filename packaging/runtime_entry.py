"""PyInstaller entrypoint; keep package imports absolute in frozen mode."""

from qwenrag_runtime.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
