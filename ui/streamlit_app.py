"""NyayaGraph web UI (SC-24).

Run the API first (python -m app.main), then:  streamlit run ui/streamlit_app.py

A pure HTTP client of the REST API in 3.4: no Neo4j, Groq or app/ imports. No login (3.2).
Case text, questions and answers live only in this browser session's memory and are
never logged (R-04). Confidence and the disclaimer are shown exactly as the API returns
them (R-24, R-25); the UI adds no legal content of its own.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `streamlit run ui/...` puts only ui/ on sys.path
    sys.path.insert(0, str(ROOT))

from ui.api_client import (  # noqa: E402
    QUERY_MODES,
    TERMINAL_JOB_STATUSES,
    UPLOAD_EXTENSIONS,
    ApiClient,
    ApiError,
    ApiUnreachable,
)
from ui.rate_budget import seconds_until_ready, tokens_used  # noqa: E402

PROJECT_NAME = "NyayaGraph"
PROJECT_TITLE = "A Multimodal Dual Knowledge GraphRAG for Consumer Legal Assistance in India"

st.set_page_config(page_title=f"{PROJECT_NAME}: Consumer Legal Assistance", layout="wide")

# Compact sidebar: Streamlit's default header strip and 1rem gaps push the settings below
# the fold on a laptop screen.
COMPACT_CSS = """
<style>
[data-testid="stSidebarHeader"] { height: 0.5rem; padding: 0; min-height: 0; }
[data-testid="stSidebarUserContent"] { padding-top: 0.5rem; padding-bottom: 1rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.8rem; }
[data-testid="stSidebar"] h1 { padding: 0 0 0.25rem 0; font-size: 1.9rem; }
[data-testid="stSidebar"] h3 { padding: 1.1rem 0 0.35rem 0; }
[data-testid="stSidebar"] [data-testid="stForm"] { padding: 0.75rem; }
/* The "Press Enter to submit form · 0/200" hint overlaps the text in the narrow sidebar. */
[data-testid="stSidebar"] [data-testid="InputInstructions"] { display: none; }
</style>
"""

MODE_LABELS = {
    "auto": "Auto (let the planner decide)",
    "case_and_legal": "My case + the law",
    "legal_only": "The law only",
    "case_only": "My case documents only",
}
STATUS_ICON = {
    "queued": "🕓", "processing": "⚙️", "running": "⚙️", "ready": "✅",
    "ready_with_warnings": "⚠️", "succeeded": "✅", "succeeded_with_warnings": "⚠️",
    "failed": "❌", "cancelled": "🚫",
}
CONFIDENCE_COLOUR = {"high": "green", "medium": "orange", "low": "red"}
JOB_POLL_SECONDS = 5  # 3.2: at most 120 requests a minute outside query and upload
DEFAULT_CASE_CHUNKS, DEFAULT_PROVISIONS = 5, 8  # 3.4 defaults
GROQ_TPM = int(os.environ.get("UI_GROQ_TPM", "8000"))  # O-6: free-tier tokens per minute
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()<>#+!|~$])")


def md(text: object) -> str:
    """Escape API text for st.markdown.

    Model-written fields can echo case text (R-06: case text is data). Escaping stops it
    from becoming links, images (which would fetch external URLs), HTML or LaTeX.
    """
    return _MD_SPECIAL.sub(r"\\\1", str(text)).replace("\n", "  \n")


# -- state ---------------------------------------------------------------------------------
def client() -> ApiClient:
    # Tests put a fake client in session_state before the first run.
    if "api_client" not in st.session_state:
        st.session_state["api_client"] = ApiClient()
    return st.session_state["api_client"]


def state() -> dict:
    ss = st.session_state
    ss.setdefault("case_id", None)
    ss.setdefault("chats", {})      # key (case_id or "legal") -> list of turns
    ss.setdefault("threads", {})    # key -> thread_id returned by the API
    ss.setdefault("jobs", {})       # job_id -> {"case_id", "filename"}
    ss.setdefault("flash", None)
    return ss


def show_error(exc: Exception, action: str) -> None:
    if isinstance(exc, ApiUnreachable):
        st.error(f"{action} failed: {exc} Start it with `python -m app.main`.")
        return
    assert isinstance(exc, ApiError)
    hint = {
        "RATE_LIMITED": f" Try again in {exc.details.get('retry_after_seconds', 'a few')} seconds.",
        "QUERY_TIMEOUT": " The question took longer than 45 seconds. Try again or narrow it.",
        "LLM_UPSTREAM_ERROR": " The language model service failed (on the free tier this is usually the daily limit).",
        "DEPENDENCY_UNAVAILABLE": f" Check that {exc.details.get('dependency', 'Neo4j and Ollama')} is running.",
        "NO_READY_DOCUMENTS": " Wait until at least one document is ready, or ask about the law only.",
        "FILE_TOO_LARGE": "",
    }.get(exc.code, "")
    ref = f" (request {exc.request_id})" if exc.request_id else ""
    st.error(f"{md(action)} failed: **{exc.code}**: {md(exc.message)}{hint}{ref}")


# -- sidebar -------------------------------------------------------------------------------
def sidebar() -> list[dict]:
    ss = state()
    with st.sidebar:
        st.markdown(COMPACT_CSS, unsafe_allow_html=True)  # static CSS only, no API text
        st.title(PROJECT_NAME)
        st.markdown(f"**{PROJECT_TITLE}**")

        try:
            health = client().health()
            if health.get("status") == "ok":
                st.caption("Backend: all systems ok")
            else:
                bad = [k for k, v in health.get("checks", {}).items() if v != "ok"]
                st.caption(f"Backend degraded: {', '.join(bad) or 'unknown'}")
        except (ApiError, ApiUnreachable) as exc:
            show_error(exc, "Health check")
            return []

        st.subheader("Case")
        try:
            cases = [c for c in client().list_cases() if c.get("status") == "active"]
        except (ApiError, ApiUnreachable) as exc:
            show_error(exc, "Loading cases")
            cases = []

        ids = [c["case_id"] for c in cases]
        titles = {c["case_id"]: c["title"] for c in cases}
        if ss["case_id"] not in ids:
            ss["case_id"] = ids[0] if ids else None
        if ids:
            ss["case_id"] = st.selectbox(
                "Active case", ids, index=ids.index(ss["case_id"]),
                format_func=lambda cid: titles.get(cid, cid), key="case_picker",
            )
        else:
            st.info("No cases yet. Create one below.")

        with st.form("new_case", clear_on_submit=True):
            title = st.text_input("New case title", max_chars=200, placeholder="e.g. Defective phone")
            if st.form_submit_button("Create case", width="stretch"):
                if not title.strip():
                    st.warning("Enter a title.")
                else:
                    try:
                        created = client().create_case(title.strip())
                        ss["case_id"] = created["case_id"]
                        ss.pop("case_picker", None)
                        ss["flash"] = f"Created case “{created['title']}”."
                        st.rerun()
                    except (ApiError, ApiUnreachable) as exc:
                        show_error(exc, "Creating the case")

        st.subheader("Answer settings")
        st.selectbox("Mode", QUERY_MODES, format_func=MODE_LABELS.get, key="mode")
        # 3.4 options. Fewer items mean fewer Groq tokens per question (O-6) but may
        # lower recall; the Phase 4 evaluation (recall at 8, 0.865) used the defaults.
        st.slider("Case passages", 1, 10, DEFAULT_CASE_CHUNKS, key="max_case_chunks",
                  help="Passages from your documents sent to the model (default 5).")
        st.slider("Provisions", 1, 12, DEFAULT_PROVISIONS, key="max_provisions",
                  help="Statute provisions sent to the model (default 8). "
                       "Lower values use fewer tokens but may miss a provision.")
        st.toggle("Show retrieval trace", key="include_trace",
                  help="Adds search query, counts and timings to each answer.")
    return cases


# -- documents tab -------------------------------------------------------------------------
def documents_tab(case_id: str) -> None:
    ss = state()
    st.subheader("Upload documents")
    st.caption("PDF (50 MB), images PNG/JPEG/TIFF (20 MB), audio MP3/WAV/M4A/MP4/OGG (200 MB), "
               "WhatsApp chat exports and other text (5 MB).")
    with st.form("upload", clear_on_submit=True):
        files = st.file_uploader("Files", type=UPLOAD_EXTENSIONS, accept_multiple_files=True)
        language = st.text_input("Audio language (optional, ISO 639-1 such as en or hi)", max_chars=2)
        if st.form_submit_button("Upload", type="primary") and files:
            for f in files:
                try:
                    res = client().upload_document(case_id, f.name, f.getvalue(), language.strip() or None)
                except (ApiError, ApiUnreachable) as exc:
                    show_error(exc, f"Uploading {f.name}")
                    continue
                if res.get("duplicate"):
                    st.info(f"{md(f.name)} is already in this case.")
                elif res.get("job_id"):
                    ss["jobs"][res["job_id"]] = {"case_id": case_id, "filename": f.name}
                    st.success(f"{md(f.name)} queued for processing.")

    job_progress(case_id)

    st.subheader("Documents in this case")
    try:
        docs = client().list_documents(case_id)
    except (ApiError, ApiUnreachable) as exc:
        show_error(exc, "Loading documents")
        return
    if not docs:
        st.info("No documents yet.")
        return
    st.dataframe(
        [{
            "": STATUS_ICON.get(d["processing_status"], ""),
            "File": d["filename"],
            "Type": d["media_type"],
            "Status": d["processing_status"].replace("_", " "),
            "Pages": d.get("page_count"),
            "Duration (s)": round(d["duration_sec"]) if d.get("duration_sec") else None,
            "Size (KB)": round(d["size_bytes"] / 1024),
            "Error": d.get("error_code") or "",
            "Uploaded": d["uploaded_at"][:16].replace("T", " "),
        } for d in docs],
        width="stretch", hide_index=True,
    )
    if st.button("Refresh list"):
        st.rerun()


@st.fragment(run_every=JOB_POLL_SECONDS)
def job_progress(case_id: str) -> None:
    ss = state()
    mine = {jid: meta for jid, meta in ss["jobs"].items() if meta["case_id"] == case_id}
    if not mine:
        return
    st.subheader("Processing")
    for job_id, meta in list(mine.items()):
        try:
            job = client().get_job(job_id)
        except ApiError as exc:
            if exc.code == "JOB_NOT_FOUND":
                ss["jobs"].pop(job_id, None)
            continue
        except ApiUnreachable:
            st.caption("Waiting for the API…")
            return
        icon = STATUS_ICON.get(job["status"], "")
        stage = (job.get("stage") or "").replace("_", " ")
        st.progress(job["progress"] / 100, text=f"{icon} {meta['filename']}: {job['status']} · {stage}")
        for w in job.get("warnings") or []:
            st.caption(f"⚠️ {md(w)}")
        if job.get("error"):
            st.error(f"{md(meta['filename'])}: {job['error']['code']}: {md(job['error']['message'])}")
        if job["status"] in TERMINAL_JOB_STATUSES:
            ss["jobs"].pop(job_id, None)


# -- ask tab -------------------------------------------------------------------------------
@st.dialog("Provision", width="large")
def provision_dialog(provision_id: str) -> None:
    try:
        p = client().get_provision(provision_id)
    except (ApiError, ApiUnreachable) as exc:
        show_error(exc, "Loading the provision")
        return
    st.markdown(f"### {md(p['citation'])}")
    bits = [f"Status: **{p['status'].replace('_', ' ')}**"]
    if p.get("in_force_from"):
        bits.append(f"in force from {p['in_force_from']}")
    if p.get("parent"):
        bits.append(f"part of {md(p['parent']['citation'])}")
    st.caption(" · ".join(bits))
    st.text(p["text"])  # verbatim statute text, no markdown interpretation
    for note in p.get("amendment_notes") or []:
        st.info(md(note))
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Refers to**")
        for r in p.get("refers_to") or []:
            st.caption(md(r["citation"]))
    with cols[1]:
        st.markdown("**Referred to by**")
        for r in p.get("referenced_by") or []:
            st.caption(md(r["citation"]))
    with cols[2]:
        st.markdown("**Defined terms used**")
        for d in p.get("definitions") or []:
            st.caption(md(d.get("term", "")))


def source_label(src: dict) -> str:
    label = src.get("filename", "document")
    if src.get("page"):
        label += f", page {src['page']}"
    if src.get("start_sec") is not None:
        start = int(src["start_sec"])
        label += f", at {start // 60}:{start % 60:02d}"
    return label


def render_answer(ans: dict, turn_key: str) -> None:
    conf = ans.get("confidence", {})
    level = conf.get("level", "low")
    intent = ans.get("intent", "").replace("_", " ").title()
    st.markdown(f":{CONFIDENCE_COLOUR.get(level, 'gray')}-badge[Confidence: {level}] :gray-badge[{intent}]")

    if ans.get("summary"):
        st.markdown(f"**{md(ans['summary'])}**")
    if ans.get("analysis"):
        st.markdown(md(ans["analysis"]))

    facts = ans.get("relevant_facts") or []
    provisions = ans.get("applicable_provisions") or []
    if facts:
        with st.expander(f"📄 Facts from your documents ({len(facts)})", expanded=True):
            for f in facts:
                st.markdown(f"- {md(f['fact'])}")
                st.caption(f"Source: {md(source_label(f.get('source', {})))}")
    if provisions:
        with st.expander(f"⚖️ Applicable provisions ({len(provisions)})", expanded=True):
            for i, p in enumerate(provisions):
                head, btn = st.columns([5, 1])
                with head:
                    status = "" if p["status"] == "in_force" else f" · :red[{p['status']}]"
                    st.markdown(f"**{md(p['citation'])}** · _{p['relation']}_{status}")
                    st.markdown(md(p["relevance"]))
                    st.caption(f"“{md(p['excerpt'])}” (text as on {p['act_as_on']})")
                with btn:
                    if st.button("Full text", key=f"prov-{turn_key}-{i}"):
                        provision_dialog(p["provision_id"])
    for prec in ans.get("precedents") or []:
        st.markdown(f"- **{md(prec['citation'])}**: {md(prec['relevance'])}")
    gaps = ans.get("gaps_or_conflicts") or []
    if gaps:
        with st.expander(f"🔎 Gaps or conflicts ({len(gaps)})"):
            for g in gaps:
                st.markdown(f"- {md(g)}")
    if conf.get("reasons"):
        st.caption("Confidence reasons: " + md("; ".join(conf["reasons"])))
    if ans.get("trace") and st.session_state.get("include_trace"):
        with st.expander("Retrieval trace"):
            st.json(ans["trace"])
    if ans.get("disclaimer"):
        st.caption(md(ans["disclaimer"]))


def record_usage(ans: dict) -> None:
    tokens = tokens_used(ans)
    if tokens:
        usage = st.session_state.setdefault("usage", [])
        usage.append((time.monotonic(), tokens))
        del usage[:-20]


@st.fragment(run_every=1)
def cooldown_note() -> None:
    usage = st.session_state.get("usage") or []
    wait = seconds_until_ready(usage, time.monotonic(), GROQ_TPM)
    if wait > 0:
        st.caption(f"Groq free tier: next question ready in about {wait} s. "
                   "Asking sooner still works but waits for retries.")
    elif usage:
        st.caption("Ready for the next question.")


def ask_tab(case_id: Optional[str]) -> None:
    ss = state()
    mode = ss.get("mode", "auto")
    effective_case = None if mode == "legal_only" else case_id
    key = effective_case or "legal"
    chat = ss["chats"].setdefault(key, [])

    top = st.columns([4, 1])
    with top[0]:
        if effective_case is None:
            st.caption("Asking about the law only (no case selected or mode is “The law only”).")
        else:
            st.caption("Answers use this case's documents and the Legal KB. Follow-up questions keep the conversation.")
    with top[1]:
        if st.button("New conversation", width="stretch"):
            ss["chats"][key] = []
            ss["threads"].pop(key, None)
            st.rerun()

    for n, turn in enumerate(chat):
        with st.chat_message("user"):
            st.markdown(md(turn["question"]))
        with st.chat_message("assistant"):
            render_answer(turn["answer"], f"{key}-{n}")

    cooldown_note()
    question = st.chat_input("Ask a question about your case or the Consumer Protection Act")
    if not question:
        return
    question = question.strip()
    if len(question) < 3:
        st.warning("Questions need at least 3 characters.")
        return
    with st.chat_message("user"):
        st.markdown(md(question))
    with st.chat_message("assistant"):
        with st.spinner("Retrieving case facts and provisions…"):
            try:
                # The trace is always requested: its token count drives the cooldown
                # note. It is only displayed when "Show retrieval trace" is on.
                ans = client().query(
                    question[:2000], case_id=effective_case, thread_id=ss["threads"].get(key),
                    mode=mode, include_trace=True,
                    max_case_chunks=int(ss.get("max_case_chunks", DEFAULT_CASE_CHUNKS)),
                    max_provisions=int(ss.get("max_provisions", DEFAULT_PROVISIONS)),
                )
            except (ApiError, ApiUnreachable) as exc:
                show_error(exc, "Answering")
                return
        record_usage(ans)
        ss["threads"][key] = ans.get("thread_id")
        chat.append({"question": question, "answer": ans})
        render_answer(ans, f"{key}-{len(chat) - 1}")


# -- case tab ------------------------------------------------------------------------------
def case_tab(case: dict) -> None:
    ss = state()
    st.markdown(f"**Title:** {md(case['title'])}")
    st.markdown(f"**Created:** {case['created_at'][:16].replace('T', ' ')} UTC")
    st.caption(f"Case id: {case['case_id']}")
    st.divider()
    st.markdown("#### Delete this case")
    st.caption("Removes every document, chunk and extracted entity of this case and the stored files. This cannot be undone.")
    confirm = st.text_input("Type the case title to confirm", key=f"confirm-{case['case_id']}")
    if st.button("Delete case", type="primary", disabled=confirm.strip() != case["title"]):
        try:
            client().delete_case(case["case_id"])
        except (ApiError, ApiUnreachable) as exc:
            show_error(exc, "Deleting the case")
            return
        for store in ("chats", "threads"):
            ss[store].pop(case["case_id"], None)
        ss["case_id"] = None
        ss.pop("case_picker", None)
        ss["flash"] = f"Deletion of “{case['title']}” started."
        st.rerun()


# -- main ----------------------------------------------------------------------------------
def main() -> None:
    ss = state()
    cases = sidebar()
    if ss.get("flash"):
        st.toast(md(ss["flash"]))
        ss["flash"] = None

    case = next((c for c in cases if c["case_id"] == ss["case_id"]), None)
    st.title(md(case["title"]) if case else "Ask about consumer law")
    ask, docs, manage = st.tabs(["💬 Ask", "📁 Documents", "⚙️ Case"])
    with ask:
        ask_tab(case["case_id"] if case else None)
    with docs:
        if case:
            documents_tab(case["case_id"])
        else:
            st.info("Create or select a case in the sidebar to upload documents.")
    with manage:
        if case:
            case_tab(case)
        else:
            st.info("No case selected.")


main()
