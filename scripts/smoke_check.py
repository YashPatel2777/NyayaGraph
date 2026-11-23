"""Phase 0 exit check (05-roadmap.md, Phase 0).

Prints OK or FAIL for four checks: Neo4j connect, Ollama embed returns 768 floats,
torch.cuda.is_available() is True, Groq returns a completion. Exits 0 only if all pass.

Reads the 1.3 variables from the environment, or from .env in the working directory.
Output carries versions, sizes and timings only, never secrets or text content (R-04).

Run: python -m scripts.smoke_check
     python -m scripts.smoke_check --whisper   (adds the faster-whisper check, 07 T3.5)
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Callable

from dotenv import load_dotenv

# Minimum Neo4j version for CREATE VECTOR INDEX (0). Not verified, see docs/README.md open items.
NEO4J_MIN_VERSION = (5, 15)

# Fixed probe text; not case data.
EMBED_PROBE = "smoke check"
GROQ_PROBE = "Reply with the single word OK."


class CheckFailed(Exception):
    pass


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if not value:
        raise CheckFailed(f"{name} is not set")
    return value


def _parse_version(raw: str) -> tuple[int, int]:
    parts = raw.split("-")[0].split(".")
    return int(parts[0]), int(parts[1])


def check_neo4j() -> str:
    from neo4j import GraphDatabase
    from neo4j.exceptions import AuthError, ServiceUnavailable

    uri = _env("NEO4J_URI", "bolt://localhost:7687")
    user = _env("NEO4J_USER", "neo4j")
    password = _env("NEO4J_PASSWORD")
    database = _env("NEO4J_DATABASE", "neo4j")
    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            driver.verify_connectivity()
            records, _, _ = driver.execute_query(
                "CALL dbms.components() YIELD name, versions, edition "
                "RETURN versions[0] AS version, edition",
                database_=database,
            )
    except AuthError as exc:
        raise CheckFailed(f"authentication failed for user {user!r}") from exc
    except ServiceUnavailable as exc:
        raise CheckFailed(f"cannot reach {uri}") from exc

    version, edition = records[0]["version"], records[0]["edition"]
    if _parse_version(version) < NEO4J_MIN_VERSION:
        need = ".".join(map(str, NEO4J_MIN_VERSION))
        raise CheckFailed(f"Neo4j {version} is older than the required {need}")
    return f"{uri} database={database} version={version} edition={edition}"


def check_ollama() -> str:
    import httpx

    url = _env("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    model = _env("EMBED_MODEL", "nomic-embed-text")
    dim = int(_env("EMBED_DIM", "768"))
    try:
        resp = httpx.post(
            f"{url}/api/embed", json={"model": model, "input": EMBED_PROBE}, timeout=60.0
        )
    except httpx.HTTPError as exc:
        raise CheckFailed(f"cannot reach {url}") from exc
    if resp.status_code != 200:
        raise CheckFailed(f"HTTP {resp.status_code} from {url}/api/embed for model {model}")

    embeddings = resp.json().get("embeddings") or []
    if len(embeddings) != 1:
        raise CheckFailed(f"expected 1 embedding, got {len(embeddings)}")
    vector = embeddings[0]
    # R-08: exactly EMBED_DIM finite floats.
    if len(vector) != dim:
        raise CheckFailed(f"expected {dim} floats, got {len(vector)}")
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
        raise CheckFailed("vector contains non-finite or non-numeric values")
    return f"{url} model={model} dim={len(vector)} all finite"


def check_cuda() -> str:
    try:
        import torch
    except ImportError as exc:
        raise CheckFailed("torch is not installed (install the CUDA build first)") from exc
    if not torch.cuda.is_available():
        raise CheckFailed(f"torch {torch.__version__} reports CUDA unavailable (CPU-only build?)")
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    return (
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"device={props.name} vram={vram_gb:.1f}GB"
    )


def check_groq() -> str:
    from langchain_groq import ChatGroq

    model = _env("GROQ_MODEL", "llama-3.3-70b-versatile")
    llm = ChatGroq(
        model=model, api_key=_env("GROQ_API_KEY"), temperature=0, max_tokens=256, max_retries=1
    )
    try:
        reply = llm.invoke(GROQ_PROBE)
    except Exception as exc:  # report the class only; messages may echo request data
        raise CheckFailed(f"completion failed: {type(exc).__name__}") from exc
    content = reply.content if isinstance(reply.content, str) else ""
    if not content.strip():
        raise CheckFailed("empty completion")
    return f"model={model} reply_chars={len(content)}"


def check_whisper() -> str:
    """Optional (--whisper, 07 T3.5): the faster-whisper call shape on CUDA float16.

    Runs the encoder on 2 s of faint noise with vad_filter off, so the cuBLAS and cuDNN
    libraries are actually loaded (README open item). The first run downloads the model.
    """
    import numpy as np
    from faster_whisper import WhisperModel

    model_name = _env("WHISPER_MODEL", "medium")
    try:
        model = WhisperModel(model_name, device="cuda", compute_type="float16")
        audio = (np.random.default_rng(0).standard_normal(32000) * 0.01).astype(np.float32)
        segments, info = model.transcribe(audio, vad_filter=False)
        segments = list(segments)  # inference runs while the generator is consumed
    except Exception as exc:
        raise CheckFailed(f"transcribe failed: {type(exc).__name__}") from exc
    for attr in ("language", "language_probability", "duration"):
        if not hasattr(info, attr):
            raise CheckFailed(f"TranscriptionInfo has no {attr}")
    if any(not hasattr(s, "start") or not hasattr(s, "end") for s in segments):
        raise CheckFailed("segments lack start or end")
    return f"model={model_name} device=cuda compute=float16 duration={info.duration:.1f}s"


CHECKS: list[tuple[str, Callable[[], str]]] = [
    ("Neo4j connect", check_neo4j),
    ("Ollama embed 768 floats", check_ollama),
    ("torch.cuda.is_available()", check_cuda),
    ("Groq completion", check_groq),
]
OPTIONAL_CHECKS: dict[str, tuple[str, Callable[[], str]]] = {
    "--whisper": ("faster-whisper CUDA float16", check_whisper),
}


def main() -> int:
    load_dotenv()
    unknown = [a for a in sys.argv[1:] if a not in OPTIONAL_CHECKS]
    if unknown:
        print(f"unknown option(s) {unknown}; available: {sorted(OPTIONAL_CHECKS)}")
        return 2
    checks = CHECKS + [OPTIONAL_CHECKS[a] for a in sys.argv[1:]]
    failures = 0
    for name, check in checks:
        start = time.perf_counter()
        try:
            detail = check()
            status = "OK  "
        except CheckFailed as exc:
            detail, status = str(exc), "FAIL"
            failures += 1
        except Exception as exc:
            detail, status = f"unexpected {type(exc).__name__}", "FAIL"
            failures += 1
        ms = (time.perf_counter() - start) * 1000
        print(f"[{status}] {name:<28} {ms:7.0f} ms  {detail}")
    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
