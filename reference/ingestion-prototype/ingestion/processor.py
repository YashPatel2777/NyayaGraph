import os
import sys
import uuid
import json
import mimetypes
import re
from typing import Dict, Any, List, Optional, Tuple

# Graceful Logfire Initialization
try:
    import logfire
    logfire.configure(service_name="enterprise-ingestion-service")
except ImportError:
    class _MockLogfire:
        def span(self, *args, **kwargs):
            from contextlib import nullcontext
            return nullcontext()
        def _safe_print(self, prefix: str, msg: str):
            try:
                print(f"[{prefix}] {msg}")
            except UnicodeEncodeError:
                print(f"[{prefix}] {msg.encode('ascii', errors='replace').decode('ascii')}")
        def info(self, *args, **kwargs): self._safe_print("INFO", str(args[0]) if args else "")
        def warning(self, *args, **kwargs): self._safe_print("WARN", str(args[0]) if args else "")
        def error(self, *args, **kwargs): self._safe_print("ERROR", str(args[0]) if args else "")
        def configure(self, *args, **kwargs): pass
    logfire = _MockLogfire()

# Safe imports for external services (Qdrant & Embeddings)
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except ImportError:
    QdrantClient = None
    models = None

try:
    from app.config import settings
except ImportError:
    class Settings:
        QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
        QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
        QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "legal_dual_rag")
    settings = Settings()

try:
    from app.services.retrieval.embeddings import embed_texts, get_embedding_dim
except ImportError:
    def embed_texts(texts: List[str]) -> List[List[float]]:
        import hashlib
        vectors = []
        for text in texts:
            # Deterministic mock 1536-dim vector for testing/offline runs
            h = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [(h[i % len(h)] / 255.0) for i in range(1536)]
            vectors.append(vec)
        return vectors

    def get_embedding_dim() -> int:
        return 1536

# Loaders and Splitter imports with multi-path resolution
try:
    from app.ingestion.loaders.pdf import parse_pdf
    from app.ingestion.loaders.audio import parse_audio
    from app.ingestion.loaders.chat import parse_chat, is_chat_export
    from app.ingestion.loaders.html import parse_html
    from app.ingestion.loaders.text import parse_text
    from app.ingestion.loaders.office import parse_office
    from app.ingestion.chunking.splitter import chunk_and_extract, ProcessedChunk
except ImportError:
    from ingestion.loaders.pdf import parse_pdf
    from ingestion.loaders.audio import parse_audio
    from ingestion.loaders.chat import parse_chat, is_chat_export
    from ingestion.loaders.html import parse_html
    from ingestion.loaders.text import parse_text
    from ingestion.loaders.office import parse_office
    from ingestion.chunking.splitter import chunk_and_extract, ProcessedChunk


PROCESSED_DATA_DIR = "processed_data"

# Initialize Qdrant Client if available
if QdrantClient and hasattr(settings, "QDRANT_URL") and settings.QDRANT_URL:
    qdrant_client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=getattr(settings, "QDRANT_API_KEY", None),
    )
else:
    qdrant_client = None


# --- 3.1 Metadata Schema Definition and Extraction Helpers ---

def build_metadata_schema(
    kb_type: str,
    modality: str,
    court_name: Optional[str] = None,
    act_title: Optional[str] = None,
    section: Optional[str] = None,
    date: Optional[str] = None,
    page_number: Optional[int] = None,
    speaker_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Standardized metadata dictionary schema required for dual-RAG queries across distinct vector stores.
    
    Example:
    {
        "kb_type": "case",
        "modality": "audio_transcript",
        "citation_metadata": {
            "court_name": "Supreme Court of India",
            "act_title": "Indian Penal Code",
            "section": "Section 302",
            "date": "2026-01-15",
            "page_number": 4,
            "speaker_id": "SPEAKER_01"
        }
    }
    """
    return {
        "kb_type": kb_type,
        "modality": modality,
        "citation_metadata": {
            "court_name": court_name,
            "act_title": act_title,
            "section": section,
            "date": date,
            "page_number": page_number,
            "speaker_id": speaker_id,
        },
    }


def extract_citation_metadata(
    chunk_text: str,
    filename: str,
    chunk_index: int,
    modality: str,
) -> Dict[str, Any]:
    """
    Extract citation metadata (court_name, act_title, section, date, page_number, speaker_id)
    directly from chunk content and document markers.
    """
    # 1. Court Name
    court_name = None
    court_patterns = [
        r"\b(Supreme Court of India)\b",
        r"\b(Delhi High Court|Bombay High Court|Madras High Court|Calcutta High Court|Allahabad High Court)\b",
        r"\b(High Court of (?:Delhi|Bombay|Madras|Calcutta|Allahabad|Karnataka|Gujarat|Kerala|Patna|Rajasthan|Madhya Pradesh|Punjab and Haryana|[A-Z][a-z]+))\b",
        r"\b(National Company Law Tribunal)\b",
        r"\b(District and Sessions Court(?:\s+of\s+[A-Z][a-z]+)?)\b",
    ]
    for cp in court_patterns:
        m = re.search(cp, chunk_text, re.IGNORECASE)
        if m:
            court_name = m.group(1).strip()
            break

    # 2. Act Title
    act_title = None
    act_match = re.search(
        r"\b([A-Z][a-zA-Z\s]+(?:Act|Code|Constitution)(?:,\s*\d{4})?)\b", 
        chunk_text
    )
    if act_match:
        act_title = act_match.group(1).strip()

    # 3. Section / Article
    section = None
    sec_match = re.search(
        r"\b(Section\s+\d+[A-Za-z]*|Article\s+\d+[A-Za-z]*|Order\s+\d+\s+Rule\s+\d+)\b", 
        chunk_text, 
        re.IGNORECASE
    )
    if sec_match:
        section = sec_match.group(1).strip()

    # 4. Date (normalized to YYYY-MM-DD)
    date_val = None
    date_match = re.search(
        r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", 
        chunk_text
    )
    if date_match:
        raw_d = date_match.group(1).strip()
        from datetime import datetime
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d-%m-%y", "%m/%d/%Y", "%m/%d/%y"):
            try:
                dt = datetime.strptime(raw_d, fmt)
                if dt.year < 100:
                    dt = dt.replace(year=2000 + dt.year)
                date_val = dt.strftime("%Y-%m-%d")
                break
            except ValueError:
                continue
        if not date_val:
            date_val = raw_d

    # 5. Page Number (e.g. from PDF page markers: "--- Page 4 ---")
    page_number = None
    page_match = re.search(r"---\s*Page\s+(\d+)\s*---", chunk_text, re.IGNORECASE)
    if page_match:
        page_number = int(page_match.group(1))
    elif modality == "pdf":
        page_number = chunk_index + 1

    # 6. Speaker ID (e.g. from Audio transcripts: "SPEAKER_01" or chat sender)
    speaker_id = None
    speaker_match = re.search(r"\b(SPEAKER_\d+)\b", chunk_text)
    if speaker_match:
        speaker_id = speaker_match.group(1)
    elif modality == "chat_log":
        chat_sender_match = re.search(r"\]\s*([^:]+?):", chunk_text)
        if chat_sender_match:
            speaker_id = chat_sender_match.group(1).strip()

    return {
        "court_name": court_name,
        "act_title": act_title,
        "section": section,
        "date": date_val,
        "page_number": page_number,
        "speaker_id": speaker_id,
    }


def resolve_kb_type(source_type: str, file_path: str, explicit_kb_type: Optional[str] = None) -> str:
    """
    Resolve target knowledge base type ('case' or 'legal').
    """
    if explicit_kb_type:
        return "legal" if "legal" in explicit_kb_type.lower() else "case"

    combined_hint = f"{source_type} {file_path}".lower()
    if any(k in combined_hint for k in ("legal", "statute", "act", "bare_act", "precedent", "doctrine", "ipc")):
        return "legal"
    return "case"


def route_file_to_loader(file_path: str, filename: str) -> Tuple[str, str]:
    """
    Step 3.2: Route files by file extension or MIME type to their respective loaders in ingestion/loaders/.
    Returns tuple of (extracted_text, modality).
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    mime_type, _ = mimetypes.guess_type(file_path)

    # Audio files
    audio_extensions = {"mp3", "wav", "m4a", "ogg", "flac", "aac", "wma"}
    if ext in audio_extensions or (mime_type and mime_type.startswith("audio/")):
        logfire.info(f"Routing {filename} to audio loader (WhisperX & Pyannote.audio)...")
        return parse_audio(file_path), "audio_transcript"

    # PDF files
    if ext == "pdf" or mime_type == "application/pdf":
        logfire.info(f"Routing {filename} to layout-aware multimodal PDF loader...")
        return parse_pdf(file_path), "pdf"

    # Chat export files
    chat_extensions = {"chat", "whatsapp", "telegram"}
    if ext in chat_extensions:
        logfire.info(f"Routing {filename} to chat loader...")
        return parse_chat(file_path), "chat_log"

    # HTML files
    if ext in ("html", "htm") or (mime_type and "html" in mime_type):
        logfire.info(f"Routing {filename} to HTML loader...")
        return parse_html(file_path), "html"

    # Office files
    if ext in ("docx", "pptx"):
        logfire.info(f"Routing {filename} to Office loader...")
        return parse_office(file_path), "office"

    # Plain text or possible chat logs exported as .txt / .log
    if ext in ("txt", "log", "text") or (mime_type and mime_type.startswith("text/")):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                sample_snippet = f.read(2048)
            if is_chat_export(sample_snippet):
                logfire.info(f"Detected chat signatures in {filename}; routing to chat loader...")
                return parse_chat(file_path), "chat_log"
        except Exception:
            pass

        logfire.info(f"Routing {filename} to plain text loader...")
        return parse_text(file_path), "plain_text"

    raise ValueError(f"Unsupported file type for {filename} (extension: '{ext}', mime: '{mime_type}')")


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def process_file(
    file_path: str, 
    filename: str, 
    source_type: str, 
    kb_type: Optional[str] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Standardized Pipeline Execution:
    1. Route files by file extension or MIME type to their respective loaders in ingestion/loaders/.
    2. Pass extracted raw content to ingestion/chunking/splitter.py while explicitly specifying target kb_type.
    3. Attach the complete metadata_schema dictionary to each final chunk before outputting chunks
       for embedding and vector database ingestion.
    """
    resolved_kb_type = resolve_kb_type(source_type, file_path, kb_type)

    with logfire.span("Processing File", file=filename, source=source_type, kb_type=resolved_kb_type):
        try:
            # 1. Route to loader
            full_text, modality = route_file_to_loader(file_path, filename)

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename} — skipping.")
                return None

            # 2. Dual feature chunking with target kb_type graph extraction
            processed_chunks: List[ProcessedChunk] = chunk_and_extract(
                text=full_text,
                chunk_size=1500,
                kb_type=resolved_kb_type,
                extract_triples=True,
            )

            if not processed_chunks:
                logfire.warning(f"No chunks generated from {filename}.")
                return None

            # 3. Standardize and attach metadata_schema to each chunk
            final_chunks_payload: List[Dict[str, Any]] = []
            chunk_texts_for_embedding: List[str] = []

            for p_chunk in processed_chunks:
                cit_meta = extract_citation_metadata(
                    chunk_text=p_chunk.text,
                    filename=filename,
                    chunk_index=p_chunk.chunk_index,
                    modality=modality,
                )

                chunk_metadata = build_metadata_schema(
                    kb_type=resolved_kb_type,
                    modality=modality,
                    court_name=cit_meta["court_name"],
                    act_title=cit_meta["act_title"],
                    section=cit_meta["section"],
                    date=cit_meta["date"],
                    page_number=cit_meta["page_number"],
                    speaker_id=cit_meta["speaker_id"],
                )
                p_chunk.metadata = chunk_metadata

                chunk_payload = {
                    "id": p_chunk.id,
                    "text": p_chunk.text,
                    "chunk_index": p_chunk.chunk_index,
                    "kb_type": resolved_kb_type,
                    "modality": modality,
                    "citation_metadata": chunk_metadata["citation_metadata"],
                    "triples": p_chunk.triples,
                    "entities": p_chunk.entities,
                    "source": filename,
                    "source_type": source_type,
                    "metadata": chunk_metadata,
                }
                final_chunks_payload.append(chunk_payload)
                chunk_texts_for_embedding.append(p_chunk.text)

            # 4. Save processed metadata locally
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "kb_type": resolved_kb_type,
                "modality": modality,
                "chunks": final_chunks_payload,
            }
            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"Saved processed data -> {local_path}")

            # 5. Embed and index in Qdrant (with explicit dual-RAG filtering flags)
            with logfire.span("Vectorizing & Indexing"):
                embeddings = embed_texts(chunk_texts_for_embedding)

                if qdrant_client and models:
                    points = [
                        models.PointStruct(
                            id=item["id"],
                            vector=vector,
                            payload=item,
                        )
                        for item, vector in zip(final_chunks_payload, embeddings)
                    ]

                    qdrant_client.upsert(
                        collection_name=settings.QDRANT_COLLECTION,
                        points=points,
                    )
                    logfire.info(f"Indexed {len(points)} points to Qdrant from {filename}.")
                else:
                    logfire.info(
                        f"Generated embeddings for {len(final_chunks_payload)} chunks "
                        "(Qdrant indexing skipped - client or collection unconfigured)."
                    )

            return final_chunks_payload

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")
            raise e


def process_directory(dir_path: str, source_type: str, kb_type: Optional[str] = None):
    """Process every file in a directory with standardized ingestion."""
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        logfire.info(f"Found {len(files)} files in {dir_path}.")
        for filename in files:
            process_file(
                file_path=os.path.join(dir_path, filename),
                filename=filename,
                source_type=source_type,
                kb_type=kb_type,
            )


def run_universal_ingestion(
    base_dir: str, 
    explicit_source_type: Optional[str] = None, 
    wipe: bool = False,
    kb_type: Optional[str] = None,
):
    """
    Scan base_dir, map sub-folders to source types and kb_types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion.
    """
    with logfire.span("Universal Ingestion Started", base_directory=base_dir, kb_type=kb_type):

        # Wipe collection if requested
        if wipe and qdrant_client and models:
            with logfire.span("Wiping Collection"):
                if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                    qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
                    logfire.info(f"Collection '{settings.QDRANT_COLLECTION}' deleted.")

        # Recreate collection — dimension resolved at runtime after embedding model probe
        if qdrant_client and models and not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            dim = get_embedding_dim()
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logfire.info(
                f"Created collection '{settings.QDRANT_COLLECTION}' "
                f"({dim}-dim, Cosine)."
            )

        # Route to sub-folders or treat the whole dir as one source
        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true" if "true" in base_name
                    else "noisy" if "noisy" in base_name
                    else "general"
                )
            logfire.info(f"No sub-folders found — processing '{base_dir}' as '{source_type}'.")
            process_directory(base_dir, source_type, kb_type=kb_type)
        else:
            for subdir in subdirs:
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir
                )
                process_directory(os.path.join(base_dir, subdir), source_type, kb_type=kb_type)


if __name__ == "__main__":
    # Usage:
    #   python -m app.ingestion.processor DATA --wipe
    #   python -m app.ingestion.processor DATA/true_data true --kb-type case
    #   python -m ingestion.processor DATA/statutes legal --kb-type legal
    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]

    explicit_kb_type = None
    if "--kb-type" in clean_args:
        kb_idx = clean_args.index("--kb-type")
        if kb_idx + 1 < len(clean_args):
            explicit_kb_type = clean_args[kb_idx + 1]
            clean_args = clean_args[:kb_idx] + clean_args[kb_idx + 2:]

    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    run_universal_ingestion(
        target_dir, 
        explicit_source_type=explicit_type, 
        wipe=wipe_requested, 
        kb_type=explicit_kb_type
    )
    logfire.info("Ingestion job completed.")