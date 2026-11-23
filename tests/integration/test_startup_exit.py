"""The real Uvicorn process exits with code 2 when a startup check fails (1.2).

Neo4j is pointed at a closed port, so this needs no running service.
"""

from __future__ import annotations

import os
import subprocess
import sys


def test_uvicorn_exits_with_code_2_when_neo4j_unreachable() -> None:
    env = {
        **os.environ,
        "NEO4J_URI": "bolt://127.0.0.1:1",
        "LOGFIRE_TOKEN": "",
        "PYTHONIOENCODING": "utf-8",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "0"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2, proc.stderr[-2000:]
    assert "Startup aborted: Neo4j is unreachable" in proc.stderr
