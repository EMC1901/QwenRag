from __future__ import annotations

import os
import subprocess
import sys

import pytest

from qwenrag_runtime.windows_job import WindowsJob


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object is Windows-only")
def test_closing_job_terminates_assigned_child_process() -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    job = WindowsJob()
    try:
        job.assign_process(process)
        job.close()
        assert process.wait(timeout=5) is not None
    finally:
        job.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
