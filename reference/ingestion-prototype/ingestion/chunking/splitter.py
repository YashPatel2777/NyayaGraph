import os
import re
import json
import uuid
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

try:
    import logfire
except ImportError:
    class _MockLogfire:
        def span(self, *args, **kwargs):
            from contextlib import nullcontext
            return nullcontext()
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
    logfire = _MockLogfire()


# --- Pydantic Schema for Structured Output Graph Extraction ---

class EntityNode(BaseModel):
    name: str = Field(description="Name or title of the extracted entity")
    type: str = Field(
        description=(
            "Entity category. For case: parties, claims, dates, events, exhibit_items, witnesses. "
            "For legal: act_name, section_or_article, legal_principles, cited_precedents, legal_remedies."
        )
    )
    description: Optional[str] = Field(
        default=None, 
        description="Brief context or description of the entity within the chunk"
    )


class RelationalEdge(BaseModel):
    source: str = Field(description="Source entity name")
    relation: str = Field(
        description=(
            "Relational edge predicate. "
            "For case: filed_claim_against, testified_on, contradicts_exhibit, occurred_on. "
            "For legal: interprets_section, overrules_precedent, derives_authority_from, prescribes_penalty_for."
        )
    )
    target: str = Field(description="Target entity name")
    context: Optional[str] = Field(
        default=None, 
        description="Contextual phrase or justification from the chunk"
    )


class GraphExtraction(BaseModel):
    entities: List[EntityNode] = Field(default_factory=list)
    relations: List[RelationalEdge] = Field(default_factory=list)

    def to_triples_list(self) -> List[Dict[str, Any]]:
        return [
            {
                "source": r.source,
                "relation": r.relation,
                "target": r.target,
                "context": r.context or "",
            }
            for r in self.relations
        ]

    def to_entities_list(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": e.name,
                "type": e.type,
                "description": e.description or "",
            }
            for e in self.entities
        ]


class ProcessedChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    chunk_index: int
    kb_type: str  # "case" | "legal"
    triples: List[Dict[str, Any]] = Field(default_factory=list)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        return self.text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "kb_type": self.kb_type,
            "triples": self.triples,
            "entities": self.entities,
            "metadata": self.metadata,
        }


# --- System Prompts for Case vs Legal Knowledge Graph Extraction ---

CASE_EXTRACTION_PROMPT = """You are an expert legal knowledge graph extractor for dual-RAG systems.
Analyze the provided courtroom, deposition, or case law text chunk and extract entities and relational edges.

TARGET ENTITIES:
- parties: plaintiffs, defendants, accused, appellants, respondents, state
- claims: causes of action, charges, allegations, remedies requested
- dates: specific dates mentioned in the text (e.g., 2026-01-15)
- events: key incidents, hearings, crimes, transactions, meetings
- exhibit_items: physical evidence, marked documents (e.g. Exhibit P-1, murder weapon, deed)
- witnesses: testifying persons, experts, deponents (e.g., PW-1, eyewitness)

TARGET RELATIONAL EDGES (must strictly be one of these types):
- filed_claim_against: (party) filed_claim_against (party or claim)
- testified_on: (witness) testified_on (event or date)
- contradicts_exhibit: (witness, party, or claim) contradicts_exhibit (exhibit_item)
- occurred_on: (event) occurred_on (date)

Extract only relations directly supported by the text chunk. Do not hallucinate.
"""

LEGAL_EXTRACTION_PROMPT = """You are an expert statutory and constitutional knowledge graph extractor for dual-RAG systems.
Analyze the provided legal statute, doctrine, or precedent chunk and extract entities and relational edges.

TARGET ENTITIES:
- act_name: statutes, codes, constitutions (e.g., Indian Penal Code, Constitution of India)
- section_or_article: statutory section or constitutional article (e.g., Section 302, Article 21)
- legal_principles: legal doctrines, maxims, rights, tests (e.g., Mens Rea, Due Process)
- cited_precedents: landmark case citations, judicial authorities
- legal_remedies: punishments, penalties, bail, injunctions, damages, remedies

TARGET RELATIONAL EDGES (must strictly be one of these types):
- interprets_section: (cited_precedent or legal_principle) interprets_section (section_or_article)
- overrules_precedent: (cited_precedent) overrules_precedent (cited_precedent)
- derives_authority_from: (section_or_article or legal_principle) derives_authority_from (act_name)
- prescribes_penalty_for: (section_or_article) prescribes_penalty_for (legal_principles or legal_remedies)

Extract only relations directly supported by the text chunk. Do not hallucinate.
"""


def _heuristic_case_extraction(text: str) -> GraphExtraction:
    """Rule-based extractor for case chunks when LLM API is unavailable."""
    entities: List[EntityNode] = []
    relations: List[RelationalEdge] = []
    
    # 1. Dates
    date_matches = re.findall(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", text)
    for d in set(date_matches):
        entities.append(EntityNode(name=d, type="dates", description="Date identified in case record"))

    # 2. Exhibit items
    exhibit_matches = re.findall(r"\b(?:Exhibit|Ex\.|Mark)\s+[A-Za-z0-9-]+\b", text, re.IGNORECASE)
    for ex in set(exhibit_matches):
        entities.append(EntityNode(name=ex, type="exhibit_items", description="Physical evidence or exhibit"))

    # 3. Witnesses
    witness_matches = re.findall(r"\b(?:PW-\d+|DW-\d+|witness\s+[A-Z][a-z]+)\b", text, re.IGNORECASE)
    for w in set(witness_matches):
        entities.append(EntityNode(name=w, type="witnesses", description="Witness or deponent"))

    # 4. Parties
    party_matches = re.findall(
        r"\b(?:Appellant|Respondent|Petitioner|Defendant|Plaintiff|State of [A-Z][a-z]+)\b", 
        text
    )
    for p in set(party_matches):
        entities.append(EntityNode(name=p, type="parties", description="Case party/litigant"))

    # 5. Extract Relations
    # occurred_on
    if date_matches:
        event_match = re.search(r"\b(incident|occurrence|accident|deposition|hearing|filing)\b", text, re.IGNORECASE)
        if event_match:
            event_name = event_match.group(0).capitalize()
            entities.append(EntityNode(name=event_name, type="events", description="Key case event"))
            relations.append(
                RelationalEdge(
                    source=event_name,
                    relation="occurred_on",
                    target=date_matches[0],
                    context=f"{event_name} occurred on {date_matches[0]}",
                )
            )

    # testified_on
    if witness_matches and date_matches:
        relations.append(
            RelationalEdge(
                source=witness_matches[0],
                relation="testified_on",
                target=date_matches[0],
                context=f"Witness testified regarding events on {date_matches[0]}",
            )
        )

    # contradicts_exhibit
    if exhibit_matches and (witness_matches or party_matches):
        subject = witness_matches[0] if witness_matches else party_matches[0]
        if re.search(r"\bcontradict|inconsistent|dispute|deny|denies\b", text, re.IGNORECASE):
            relations.append(
                RelationalEdge(
                    source=subject,
                    relation="contradicts_exhibit",
                    target=exhibit_matches[0],
                    context=f"{subject} contradicts {exhibit_matches[0]}",
                )
            )

    # filed_claim_against
    if len(party_matches) >= 2:
        relations.append(
            RelationalEdge(
                source=party_matches[0],
                relation="filed_claim_against",
                target=party_matches[1],
                context=f"{party_matches[0]} filed claim against {party_matches[1]}",
            )
        )

    return GraphExtraction(entities=entities, relations=relations)


def _heuristic_legal_extraction(text: str) -> GraphExtraction:
    """Rule-based extractor for legal/statutory chunks when LLM API is unavailable."""
    entities: List[EntityNode] = []
    relations: List[RelationalEdge] = []

    # 1. Acts
    act_matches = re.findall(
        r"\b(?:[A-Z][a-zA-Z\s]+Act(?:,\s*\d{4})?|Indian Penal Code|Constitution of India|Code of Criminal Procedure)\b",
        text
    )
    for a in set(act_matches):
        entities.append(EntityNode(name=a.strip(), type="act_name", description="Statutory enactment"))

    # 2. Section or Article
    section_matches = re.findall(r"\b(?:Section\s+\d+[A-Za-z]*|Article\s+\d+[A-Za-z]*)\b", text, re.IGNORECASE)
    for s in set(section_matches):
        entities.append(EntityNode(name=s.strip(), type="section_or_article", description="Statutory provision"))

    # 3. Legal Principles
    principles_pool = [
        "Mens Rea", "Actus Reus", "Audi Alteram Partem", "Res Judicata", "Stare Decisis",
        "Due Process", "Presumption of Innocence", "Fundamental Rights", "Right to Life"
    ]
    for p in principles_pool:
        if re.search(rf"\b{re.escape(p)}\b", text, re.IGNORECASE):
            entities.append(EntityNode(name=p, type="legal_principles", description="Jurisprudential principle"))

    # 4. Remedies / Penalties
    remedies_pool = ["Imprisonment", "Penalty", "Fine", "Bail", "Injunction", "Damages", "Acquittal", "Conviction"]
    for r in remedies_pool:
        if re.search(rf"\b{re.escape(r)}\b", text, re.IGNORECASE):
            entities.append(EntityNode(name=r, type="legal_remedies", description="Prescribed legal remedy or penalty"))

    # 5. Extract Relations
    if section_matches and act_matches:
        relations.append(
            RelationalEdge(
                source=section_matches[0],
                relation="derives_authority_from",
                target=act_matches[0],
                context=f"{section_matches[0]} enacted under {act_matches[0]}",
            )
        )

    if section_matches and any(e.type == "legal_remedies" for e in entities):
        remedy = next(e.name for e in entities if e.type == "legal_remedies")
        relations.append(
            RelationalEdge(
                source=section_matches[0],
                relation="prescribes_penalty_for",
                target=remedy,
                context=f"{section_matches[0]} prescribes {remedy}",
            )
        )

    if section_matches and any(e.type == "legal_principles" for e in entities):
        principle = next(e.name for e in entities if e.type == "legal_principles")
        relations.append(
            RelationalEdge(
                source=principle,
                relation="interprets_section",
                target=section_matches[0],
                context=f"{principle} interprets {section_matches[0]}",
            )
        )

    return GraphExtraction(entities=entities, relations=relations)


def extract_graph_triples(
    chunk_text: str,
    kb_type: str = "case",
    model: str = "gpt-4o-mini",
) -> GraphExtraction:
    """
    Call extraction LLM using structured output parsing to identify entity nodes
    and relational edges within chunk bounds.
    
    Target entity and relationship extraction:
    - Case chunks extraction target:
        * Entity nodes: parties, claims, dates, events, exhibit items, witnesses.
        * Relational edges: filed_claim_against, testified_on, contradicts_exhibit, occurred_on.
    - Legal chunks extraction target:
        * Entity nodes: act name, section or article number, legal principles, cited precedents, legal remedies.
        * Relational edges: interprets_section, overrules_precedent, derives_authority_from, prescribes_penalty_for.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    kb_type_norm = kb_type.lower().strip()
    is_legal = "legal" in kb_type_norm or "statute" in kb_type_norm or "act" in kb_type_norm

    if not api_key:
        # Fall back to heuristic rule-based extraction
        logfire.info(f"OpenAI API key not set, using heuristic graph extractor for kb_type='{kb_type}'")
        if is_legal:
            return _heuristic_legal_extraction(chunk_text)
        return _heuristic_case_extraction(chunk_text)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        system_prompt = LEGAL_EXTRACTION_PROMPT if is_legal else CASE_EXTRACTION_PROMPT

        # Use OpenAI structured output parsing
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Extract knowledge graph entities and relations from this chunk:\n\n{chunk_text}",
                },
            ],
            response_format=GraphExtraction,
            temperature=0.0,
        )
        extraction: GraphExtraction = completion.choices[0].message.parsed
        return extraction

    except Exception as e:
        logfire.warning(f"LLM structured graph extraction failed ({e}), falling back to heuristic.")
        if is_legal:
            return _heuristic_legal_extraction(chunk_text)
        return _heuristic_case_extraction(chunk_text)


def chunk_and_extract(
    text: str,
    chunk_size: int = 1500,
    kb_type: str = "case",
    extract_triples: bool = True,
) -> List[ProcessedChunk]:
    """
    Dual feature extraction during chunking:
    Generates structured graph triples alongside standard text embeddings for each text chunk.
    """
    with logfire.span("✂️ Dual Chunking & Graph Triple Extraction", text_length=len(text), kb_type=kb_type):
        if not text.strip():
            return []

        paragraphs = text.split("\n\n")
        raw_chunks: List[str] = []
        current_chunk = ""

        for p in paragraphs:
            if len(current_chunk) + len(p) < chunk_size:
                current_chunk += p + "\n\n"
            else:
                if current_chunk.strip():
                    raw_chunks.append(current_chunk.strip())
                current_chunk = p + "\n\n"

        if current_chunk.strip():
            raw_chunks.append(current_chunk.strip())

        valid_raw_chunks = [c for c in raw_chunks if c.strip()]
        processed_chunks: List[ProcessedChunk] = []

        for idx, chunk_str in enumerate(valid_raw_chunks):
            triples: List[Dict[str, Any]] = []
            entities: List[Dict[str, Any]] = []

            if extract_triples:
                extraction = extract_graph_triples(chunk_str, kb_type=kb_type)
                triples = extraction.to_triples_list()
                entities = extraction.to_entities_list()

            processed_chunks.append(
                ProcessedChunk(
                    text=chunk_str,
                    chunk_index=idx,
                    kb_type=kb_type,
                    triples=triples,
                    entities=entities,
                )
            )

        logfire.info(
            f"✅ Generated {len(processed_chunks)} chunks with "
            f"{sum(len(c.triples) for c in processed_chunks)} extracted graph triples."
        )
        return processed_chunks


def chunk_text(
    text: str, 
    chunk_size: int = 1500,
    extract_graph: bool = False,
    kb_type: str = "case",
) -> Union[List[str], List[ProcessedChunk]]:
    """
    Text chunker with dual-feature support.
    If extract_graph is False (default for backwards compatibility), returns List[str].
    If extract_graph is True, returns List[ProcessedChunk] containing both text and graph triples.
    """
    if extract_graph:
        return chunk_and_extract(text, chunk_size=chunk_size, kb_type=kb_type, extract_triples=True)

    # Standard string chunking for backward compatibility
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            if len(current_chunk) + len(p) < chunk_size:
                current_chunk += p + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = p + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")
        return valid_chunks