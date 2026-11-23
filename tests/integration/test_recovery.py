"""Restart-recovery test (Phase 6, E-11): kill the server process in the middle of a job; on
restart the job is re-queued and finishes.

Opt-in (`pytest -m process`, A-46): it runs a real Uvicorn process with the GPU models, kills
it abruptly (`taskkill /F`, like a power cut) while a voice-note job is running, starts it
again, and checks that the job finishes with attempts 1 (one E-11 re-queue; a new job starts at
0) and the document ready. The server
runs against its own throwaway database and upload folder, on a free loopback port.

ServerProcess is shared with the soak test.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from app.services.retrieval.neo4j_service import Neo4jService
from scripts.init_neo4j import init_schema
from tests.integration.test_case_processor import FIXTURES

pytestmark = [pytest.mark.process, pytest.mark.neo4j, pytest.mark.ollama, pytest.mark.groq, pytest.mark.gpu]

ROOT = Path(__file__).resolve().parents[2]
TERMINAL = {"succeeded", "succeeded_with_warnings", "failed", "cancelled"}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerProcess:
    """A real Uvicorn process for the app, bound to 127.0.0.1 (APP_HOST) on a free port."""

    def __init__(self, log: Path, env: dict[str, str] | None = None) -> None:
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.log = log
        self.env = {**os.environ, **(env or {})}
        self.proc: subprocess.Popen | None = None

    def start(self, timeout: float = 240.0) -> None:
        with self.log.open("a", encoding="utf-8") as fh:
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(self.port)],
                cwd=ROOT, env=self.env, stdout=fh, stderr=subprocess.STDOUT,
            )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with code {self.proc.returncode}; see {self.log}")
            try:
                if httpx.get(f"{self.base}/health", timeout=5).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)
        raise TimeoutError(f"server not healthy within {timeout} s; see {self.log}")

    def kill(self) -> None:
        """Abrupt stop, like a power cut (A-46): no shutdown handlers run."""
        assert self.proc is not None
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], capture_output=True, check=False)
        self.proc.wait(timeout=30)

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.kill()


def _job(base: str, job_id: str) -> dict:
    return httpx.get(f"{base}/v1/jobs/{job_id}", timeout=10).json()


def test_killed_mid_job_resumes_and_finishes(neo4j_test_db: Neo4jService, tmp_path: Path) -> None:
    init_schema(neo4j_test_db)
    env = {"NEO4J_DATABASE": neo4j_test_db.database, "UPLOAD_DIR": str(tmp_path / "uploads")}
    server = ServerProcess(tmp_path / "server.log", env)
    server.start()
    try:
        case_id = httpx.post(f"{server.base}/v1/cases", json={"title": "recovery test"}, timeout=10).json()["case_id"]
        wav = FIXTURES / "clean" / "voice_note.wav"
        up = httpx.post(f"{server.base}/v1/cases/{case_id}/documents",
                        files={"file": (wav.name, wav.read_bytes(), "audio/wav")}, timeout=30)
        assert up.status_code == 202, up.text
        job_id = up.json()["job_id"]

        # Kill as soon as the job is running and not yet done.
        deadline = time.monotonic() + 120
        while True:
            job = _job(server.base, job_id)
            if job["status"] == "running" and job["stage"] != "done":
                break
            assert job["status"] not in TERMINAL, f"job finished before the kill: {job['status']}"
            assert time.monotonic() < deadline, "job never started running"
            time.sleep(0.2)
        killed_at_stage = job["stage"]
        server.kill()

        row = neo4j_test_db.read_sync("MATCH (j:Job {id: 'job:' + $id}) RETURN j.status AS status", {"id": job_id})
        assert row == [{"status": "running"}], row  # left running by the crash

        server.start()  # E-11 recovery runs at startup
        deadline = time.monotonic() + 600
        while (job := _job(server.base, job_id))["status"] not in TERMINAL:
            assert time.monotonic() < deadline, f"job did not finish after the restart: {job}"
            time.sleep(1)
        assert job["status"] in ("succeeded", "succeeded_with_warnings"), job
        attempts = neo4j_test_db.read_sync("MATCH (j:Job {id: 'job:' + $id}) RETURN j.attempts AS a", {"id": job_id})
        assert attempts == [{"a": 1}], attempts  # E-11: one re-queue (a new job starts at 0)
        doc = httpx.get(f"{server.base}/v1/cases/{case_id}/documents/{up.json()['doc_id']}", timeout=10).json()
        assert doc["processing_status"] in ("ready", "ready_with_warnings"), doc
        print(f"\nrecovery: killed at stage {killed_at_stage!r}; job {job['status']} after restart, attempts 1")
    finally:
        server.stop()
