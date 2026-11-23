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

def parse_office(file_path: str):
    """
    Parses Office documents (.docx, .pptx) using the Unstructured library.
    Unlike PDFs, these formats are structured and lightweight, so they are processed locally.
    """
    with logfire.span("📄 Office Document Parsing", filename=file_path):
        try:
            from unstructured.partition.auto import partition
            elements = partition(filename=file_path)
            full_text = "\n".join([str(el) for el in elements])
            
            if not full_text.strip():
                logfire.warning(f"⚠️ Unstructured returned empty text for {file_path}")
            else:
                logfire.info(f"✅ Successfully parsed {len(full_text)} characters")

            return full_text
        except Exception as e:
            logfire.error(f"❌ Office Parse Failed: {e}")
            raise e
