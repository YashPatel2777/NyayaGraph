# NyayaGraph setup (Windows 11, no Docker)

From a clean Windows machine to a passing smoke check and a first `POST /v1/query`. Read `README.md` first for how the project is organised. The specification is in `docs/`.

## 1. What you need

| Component | Version or note |
| --- | --- |
| Windows 11 | The host OS for v1 (section 0) |
| NVIDIA GPU and driver | Tested on an RTX 3050 Laptop GPU with 4 GB (D-2). EasyOCR and Whisper `medium` fit together |
| Python | 3.11 (tested with 3.11.5) |
| Neo4j Desktop | DBMS version **2026.10.2**, required by SC-15. Desktop runs the enterprise edition, which the tests need for throwaway databases |
| Ollama | Native Windows app, serving `nomic-embed-text` |
| Groq API key | The free tier works but is limited to 8,000 tokens per minute and **200,000 tokens per day** (O-6) |

There is no Docker in this project.

## 2. Neo4j

1. Install Neo4j Desktop and create a local DBMS with version **2026.10.2**. Set a password, which goes into `NEO4J_PASSWORD` below.
2. Start the DBMS. The app connects to `bolt://localhost:7687`, database `neo4j`.
3. Nothing else is installed into Neo4j. The schema is created by `scripts.init_neo4j` (step 6).

## 3. Ollama

1. Install Ollama for Windows and let it run (tray app), or run `ollama serve`.
2. Pull the embedding model, the only one the project uses (R-08):

   ```
   ollama pull nomic-embed-text
   ```

## 4. Python environment

From the project folder (for example `G:\DJSCE\MP\NyayaGraph`):

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Install torch and torchvision **first, together, from the CUDA index** (D-3). `requirements.txt` deliberately does not list them. Installing them any other way lets EasyOCR pull CPU-only builds from PyPI.

Models are downloaded the first time they are used and cached afterwards:
- the FlashRank reranker `ms-marco-MiniLM-L-12-v2` (SC-21) goes to `\tmp` on the current drive (A-32);
- Whisper `medium` and the EasyOCR models go to their libraries' default caches.

## 5. Configuration (`.env`)

Copy `.env.example` to `.env` and fill in the two required values. Never commit `.env`.

| Variable | Default | Notes |
| --- | --- | --- |
| `APP_ENV` | `dev` | `prod` turns the `/docs` pages off |
| `APP_HOST` | `127.0.0.1` | Keep it loopback: v1 has no authentication (3.2) |
| `DEFAULT_USER_ID` | `local` | The single user in v1 |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_DATABASE` | `bolt://localhost:7687`, `neo4j`, `neo4j` | |
| `NEO4J_PASSWORD` | **required** | The DBMS password from step 2 |
| `OLLAMA_URL` | `http://localhost:11434` | |
| `EMBED_MODEL`, `EMBED_DIM` | `nomic-embed-text`, `768` | Locked; checked against the database's Meta node at startup |
| `GROQ_API_KEY` | **required** | |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | D-1 (the spec's original model is no longer served) |
| `FLASHRANK_MODEL` | `ms-marco-MiniLM-L-12-v2` | SC-21; empty means the library default (weaker, O-7) |
| `WHISPER_MODEL`, `OCR_LANGS` | `medium`, `en,hi` | |
| `UPLOAD_DIR` | `./data/uploads` | Case uploads; gitignored |
| `ENABLE_GUARDRAILS` | `false` | Guardrails are out of scope in v1 |
| `LINK_MIN_SCORE`, `RERANK_HIGH` | `0.60`, `0.5` | Initial values, to be calibrated |
| `CORS_ORIGINS` | empty | Empty means CORS is off |
| `LOGFIRE_TOKEN` | empty | Optional tracing; scrubbing stays on (R-04) |

## 6. Create the schema

```
python -m scripts.init_neo4j
```

It creates the constraints, the vector and full-text indexes and the Meta node (2.2). Running it again changes nothing. On a database created before SC-15 it rebuilds the case vector index once.

## 7. Check the environment

```
python -m scripts.smoke_check
```

All four checks must print `OK`: Neo4j connect, Ollama returns 768 floats, `torch.cuda.is_available()`, and a Groq completion. Add `--whisper` to also load EasyOCR and Whisper on the GPU.

## 8. Load the Legal KB

The statutes are in `data/legal/`, each with its ingest metadata next to it (A-47). Ingest them **in this order**, because the rules refer to the Act and the amendment needs its principal rules:

```
python -m scripts.ingest_legal data/legal/CPA2019.pdf --act-id cpa-2019 --metadata data/legal/CPA2019.metadata.json
python -m scripts.ingest_legal "data/legal/Consumer Protection (Jurisdiction of the District Commission, the State Commission and the National Commission) Rules, 2021.pdf" --act-id cp-jurisdiction-rules-2021 --metadata "data/legal/Consumer Protection (Jurisdiction of the District Commission, the State Commission and the National Commission) Rules, 2021.metadata.json"
python -m scripts.ingest_legal "data/legal/E commerce rules.pdf" --act-id cp-e-commerce-rules-2020 --metadata "data/legal/E commerce rules.metadata.json"
python -m scripts.ingest_legal "data/legal/2021 May 17.pdf" --act-id cp-e-commerce-amendment-rules-2021 --metadata "data/legal/2021 May 17.metadata.json"
```

Expected results:
- **CPA 2019:** 107 sections and 47 definitions.
- **Jurisdiction Rules:** 5 rules and 3 PRESCRIBES_FOR edges.
- **E-Commerce Rules:** 8 rules.
- **Amendment:** substitutes rule 4(1) of the E-Commerce Rules.

The golden tests check the same counts in a throwaway database: `pytest tests/golden -q`.

Two files in `data/legal/` are deliberately **not** ingested:
- `COI.pdf`, the Constitution (D-5);
- `Consumer Protection (E-Commerce) Rules, 2020.pdf`, a poor scan replaced by the Gazette copy (SC-8).

## 9. Run the service

```
python -m app.main
```

It binds `APP_HOST` (127.0.0.1) on port 8000. The equivalent command is `uvicorn app.main:app --host 127.0.0.1 --port 8000`.

At startup it checks Neo4j, Ollama, the indexes and the Meta node, loads OCR, Whisper and the reranker, and resumes interrupted jobs (E-11). If a startup check fails, the process **exits with code 2** and prints the reason.

Useful endpoints:
- `GET http://127.0.0.1:8000/health`: 200 when all dependencies are fine, 503 with per-check detail otherwise;
- `http://127.0.0.1:8000/docs`: the OpenAPI page (dev only);
- `POST /v1/cases`, then `POST /v1/cases/{case_id}/documents` (multipart field `file`), then `POST /v1/query` with `{"question": "...", "case_id": "..."}`.

### Web UI (optional, SC-24)

With the service running, in a second terminal (same virtual environment):

```
pip install -r requirements-ui.txt
streamlit run ui/streamlit_app.py
```

Run it from the project root so `.streamlit/config.toml` is read. It opens on `http://127.0.0.1:8501`. The sidebar shows backend health and lets you create or pick a case; the tabs let you ask questions, upload documents and watch them process, and delete a case. Click **Full text** on a provision to see the statute text and its links. If the API runs elsewhere, set `UI_API_BASE_URL` (default `http://127.0.0.1:8000`).

## 10. Tests

| Command | What runs | Needs |
| --- | --- | --- |
| `pytest -q` | Everything except the opt-in tests, about 6 minutes | Neo4j (throwaway databases), Ollama, Groq for a few tests, GPU for the media tests |
| `pytest tests/unit -q` | Unit tests only, seconds | Nothing running |
| `pytest tests/golden -q` | Statute parsing checks G1 to G12 | Neo4j and Ollama; it ingests the PDFs into its own throwaway database |
| `pytest -m soak` | Phase 6 soak test: 20 concurrent queries and one upload against a real server | Everything above; spends Groq quota |
| `pytest -m process` | Phase 6 restart-recovery test: kills the server mid-job | Everything above; GPU and Groq |
| `python -m tests.eval.run_eval` | The 31-question evaluation (recall at 8, citations, latency) | The four synthetic cases loaded; about 130,000 Groq tokens |

## 11. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Startup exits with code 2: 'indexes not ONLINE' or 'lacks filter properties' | Run `python -m scripts.init_neo4j` (SC-15) |
| Startup exits with code 2: Meta mismatch | The database was built with another embedding model. Use a new database or re-embed (2.2) |
| Queries return 504 QUERY_TIMEOUT; Groq 429 'tokens per day' | The free-tier daily budget (200,000) is used up. It rolls over within 24 hours (O-6) |
| 429 RATE_LIMITED from the API | Per-IP limits: 30 queries, 10 uploads, 120 other requests a minute (3.2) |
| Whisper falls back to CPU, or CUDA DLL errors | Check the torch install from step 4 and run `python -m scripts.smoke_check --whisper` |
| Throwaway-database tests are skipped | Neo4j Community has one user database. Use Neo4j Desktop (enterprise) |
