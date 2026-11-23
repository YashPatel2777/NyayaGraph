import os
import io
import re
import base64
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from PIL import Image

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


@dataclass
class VisualRegionSummary:
    page_number: int
    region_type: str  # "stamp", "signature", "receipt", "evidence", "image"
    summary_text: str
    bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass
class PDFPageContent:
    page_number: int
    text_paragraphs: List[str] = field(default_factory=list)
    markdown_tables: List[str] = field(default_factory=list)
    visual_summaries: List[VisualRegionSummary] = field(default_factory=list)

    def render(self) -> str:
        """Render page content into structured layout-aware markdown."""
        parts = [f"--- Page {self.page_number} ---"]
        
        # 1. Text paragraphs
        if self.text_paragraphs:
            parts.append("\n\n".join(self.text_paragraphs))
            
        # 2. Tabular structures rendered in Markdown
        if self.markdown_tables:
            parts.append("#### Tabular Structures:")
            parts.append("\n\n".join(self.markdown_tables))
            
        # 3. Visual image regions (stamps, signatures, receipts)
        if self.visual_summaries:
            parts.append("#### Visual Regions & Forensics:")
            for vs in self.visual_summaries:
                parts.append(f"- [{vs.region_type.upper()} SUMMARY - Page {vs.page_number}]: {vs.summary_text}")
                
        return "\n\n".join(parts)


@dataclass
class PDFMultimodalResult:
    full_text: str
    pages: List[PDFPageContent]
    total_pages: int
    has_tables: bool
    has_visual_elements: bool


def _table_to_markdown(table_data: List[List[Any]]) -> str:
    """Convert a 2D table grid into a standard Markdown table."""
    if not table_data or not any(table_data):
        return ""

    # Clean cells
    cleaned_rows: List[List[str]] = []
    max_cols = 0
    for row in table_data:
        cleaned_row = [
            str(cell).strip().replace("\n", " ").replace("|", "\\|") if cell is not None else ""
            for cell in row
        ]
        if any(cleaned_row):
            cleaned_rows.append(cleaned_row)
            max_cols = max(max_cols, len(cleaned_row))

    if not cleaned_rows or max_cols == 0:
        return ""

    # Pad rows to max_cols
    for row in cleaned_rows:
        while len(row) < max_cols:
            row.append("")

    headers = cleaned_rows[0]
    separator = ["---"] * max_cols
    body_rows = cleaned_rows[1:] if len(cleaned_rows) > 1 else []

    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body_rows:
        md_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(md_lines)


def process_visual_region_with_vision_llm(
    image_bytes: bytes,
    page_number: int,
    region_hint: str = "stamp or signature",
) -> str:
    """
    Process visual image regions containing physical stamps, seals, signatures, or receipts
    using a Vision LLM (OpenAI GPT-4o / GPT-4o-mini) to produce a concise textual visual summary.
    Falls back gracefully if API key is not configured.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return f"Physical visual region ({region_hint}) identified on page {page_number}. (Vision LLM key not configured for forensic extraction)."

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        base64_img = base64.b64encode(image_bytes).decode("utf-8")
        prompt = (
            "You are a forensic legal document analyzer. Analyze this visual region extracted from a "
            "legal contract, court document, or physical evidence. Identify and summarize any physical stamps, "
            "notary seals, embossed emblems, signatures, handwriting, or receipt figures. "
            "Return a concise 1-2 sentence textual visual summary."
        )

        response = client.chat.completions.create(
            model=os.getenv("VISION_LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_img}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
            max_tokens=150,
            temperature=0.1,
        )
        summary = response.choices[0].message.content
        return summary.strip() if summary else "Unspecified visual element detected."
    except Exception as e:
        logfire.warning(f"Vision LLM extraction failed on page {page_number}: {e}")
        return f"Visual region detected ({region_hint}) on page {page_number}."


def _parse_with_docling(file_path: str) -> Optional[PDFMultimodalResult]:
    """Integration for Docling document layout analysis framework."""
    try:
        from docling.document_converter import DocumentConverter
        logfire.info("Attempting layout parsing using Docling framework...")
        converter = DocumentConverter()
        conversion_res = converter.convert(file_path)
        doc = conversion_res.document
        
        # Export structured markdown
        md_text = doc.export_to_markdown()
        if md_text and md_text.strip():
            logfire.info("Docling layout analysis completed successfully.")
            return PDFMultimodalResult(
                full_text=md_text,
                pages=[],
                total_pages=len(getattr(doc, "pages", [])) or 1,
                has_tables="|" in md_text,
                has_visual_elements=True,
            )
    except ImportError:
        pass
    except Exception as e:
        logfire.warning(f"Docling parsing encountered error: {e}")
    return None


def _parse_with_unstructured(file_path: str) -> Optional[PDFMultimodalResult]:
    """Integration for Unstructured document layout analysis framework."""
    try:
        from unstructured.partition.pdf import partition_pdf
        logfire.info("Attempting layout parsing using Unstructured framework...")
        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=True,
        )
        
        page_dict: Dict[int, PDFPageContent] = {}
        for el in elements:
            pg = getattr(el.metadata, "page_number", 1) or 1
            if pg not in page_dict:
                page_dict[pg] = PDFPageContent(page_number=pg)
            
            el_type = type(el).__name__
            el_text = str(el).strip()
            if not el_text:
                continue

            if "Table" in el_type:
                # Check for table HTML or markdown
                table_html = getattr(el.metadata, "text_as_html", None)
                page_dict[pg].markdown_tables.append(table_html or el_text)
            elif "Image" in el_type or "Figure" in el_type:
                page_dict[pg].visual_summaries.append(
                    VisualRegionSummary(
                        page_number=pg,
                        region_type="stamp_or_evidence",
                        summary_text=el_text,
                    )
                )
            else:
                page_dict[pg].text_paragraphs.append(el_text)

        pages = [page_dict[k] for k in sorted(page_dict.keys())]
        full_text = "\n\n".join(p.render() for p in pages)
        logfire.info("Unstructured layout analysis completed successfully.")
        return PDFMultimodalResult(
            full_text=full_text,
            pages=pages,
            total_pages=len(pages),
            has_tables=any(p.markdown_tables for p in pages),
            has_visual_elements=any(p.visual_summaries for p in pages),
        )
    except ImportError:
        pass
    except Exception as e:
        logfire.warning(f"Unstructured parsing encountered error: {e}")
    return None


def _parse_with_pymupdf_multimodal(file_path: str) -> PDFMultimodalResult:
    """
    High-fidelity layout-aware multimodal PDF parser using PyMuPDF (fitz) + Vision LLM.
    1. Extracts layout text paragraphs preserving semantic reading order.
    2. Identifies tabular structures and formats them in clean Markdown tables.
    3. Detects visual image regions (stamps, signatures, receipts) and calls Vision LLM
       to generate textual visual summaries.
    """
    import fitz  # PyMuPDF
    
    doc = fitz.open(file_path)
    total_pages = len(doc)
    page_contents: List[PDFPageContent] = []
    has_tables_total = False
    has_visuals_total = False

    logfire.info(f"PyMuPDF layout analysis active on {total_pages} pages for {file_path}...")

    for page_idx in range(total_pages):
        page = doc[page_idx]
        page_num = page_idx + 1
        page_content = PDFPageContent(page_number=page_num)

        # --- 1. Tabular Structure Extraction ---
        table_rects: List[fitz.Rect] = []
        try:
            tabs = page.find_tables()
            if tabs and tabs.tables:
                for tab in tabs.tables:
                    raw_data = tab.extract()
                    md_table = _table_to_markdown(raw_data)
                    if md_table:
                        page_content.markdown_tables.append(md_table)
                        table_rects.append(tab.bbox)
                        has_tables_total = True
        except Exception as tab_err:
            logfire.warning(f"Page {page_num} table extraction note: {tab_err}")

        # --- 2. Text Paragraphs Extraction ---
        # Get blocks: (x0, y0, x1, y1, text, block_no, block_type)
        # block_type 0 = text, 1 = image
        blocks = page.get_text("blocks")
        for b in blocks:
            b_rect = fitz.Rect(b[0], b[1], b[2], b[3])
            b_type = b[6]
            b_text = b[4].strip()

            if b_type == 0 and b_text:
                # Check if block overlaps inside an already extracted table
                is_inside_table = any(b_rect.intersects(t_rect) for t_rect in table_rects)
                if not is_inside_table:
                    page_content.text_paragraphs.append(b_text)

        # --- 3. Visual Regions Extraction (Stamps, Signatures, Evidence) ---
        image_list = page.get_images(full=True)
        # Inspect visual candidates
        for img_info in image_list:
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image")
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # Filter out tiny icon artifacts (e.g. icons smaller than 50x50)
                if width < 50 or height < 50 or not image_bytes:
                    continue

                # Determine region hint based on page placement or dimensions
                region_hint = "stamp_or_signature"
                if width > 350 and height > 250:
                    region_hint = "physical_evidence_or_receipt"

                # Send image bytes to Vision LLM for concise textual summary
                visual_summary = process_visual_region_with_vision_llm(
                    image_bytes=image_bytes,
                    page_number=page_num,
                    region_hint=region_hint,
                )

                page_content.visual_summaries.append(
                    VisualRegionSummary(
                        page_number=page_num,
                        region_type=region_hint,
                        summary_text=visual_summary,
                    )
                )
                has_visuals_total = True

            except Exception as img_err:
                logfire.warning(f"Error inspecting image xref {xref} on page {page_num}: {img_err}")

        page_contents.append(page_content)

    doc.close()

    full_rendered_text = "\n\n".join(p.render() for p in page_contents)
    return PDFMultimodalResult(
        full_text=full_rendered_text,
        pages=page_contents,
        total_pages=total_pages,
        has_tables=has_tables_total,
        has_visual_elements=has_visuals_total,
    )


def _parse_with_pypdf_fallback(file_path: str) -> str:
    """Fallback using pypdf if advanced layout engines fail."""
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    text_parts: List[str] = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        if t.strip():
            text_parts.append(f"--- Page {i + 1} ---\n{t.strip()}")
    return "\n\n".join(text_parts)


def parse_pdf(file_path: str) -> str:
    """
    Layout-aware multimodal parsing for scanned legal contracts, physical evidence, stamps, and receipts.
    
    1. Integrates Docling or Unstructured framework for PDF layout analysis.
    2. Separates PDF content into text paragraphs, tabular structures rendered in Markdown,
       and image regions containing physical stamps or signatures.
    3. Processes visual image regions with a vision LLM to produce concise textual visual summaries.
    
    Returns structured layout-aware multimodal markdown text.
    """
    with logfire.span("📑 Multimodal PDF Parsing", filename=file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        # 1. Try Docling framework
        docling_result = _parse_with_docling(file_path)
        if docling_result and docling_result.full_text.strip():
            return docling_result.full_text

        # 2. Try Unstructured framework
        unstructured_result = _parse_with_unstructured(file_path)
        if unstructured_result and unstructured_result.full_text.strip():
            return unstructured_result.full_text

        # 3. High-fidelity PyMuPDF multimodal layout analysis with Vision LLM
        try:
            pymupdf_result = _parse_with_pymupdf_multimodal(file_path)
            if pymupdf_result and pymupdf_result.full_text.strip():
                logfire.info(
                    f"✅ Layout parsing extracted {len(pymupdf_result.full_text)} characters "
                    f"across {pymupdf_result.total_pages} pages."
                )
                return pymupdf_result.full_text
        except Exception as e:
            logfire.warning(f"PyMuPDF layout parsing failed, using pypdf fallback: {e}")

        # 4. Fallback to basic text parser
        fallback_text = _parse_with_pypdf_fallback(file_path)
        return fallback_text


def parse_pdf_multimodal(file_path: str) -> PDFMultimodalResult:
    """
    Returns structured PDFMultimodalResult with separated pages, tables, and visual forensics.
    """
    docling_result = _parse_with_docling(file_path)
    if docling_result:
        return docling_result

    unstructured_result = _parse_with_unstructured(file_path)
    if unstructured_result:
        return unstructured_result

    return _parse_with_pymupdf_multimodal(file_path)
