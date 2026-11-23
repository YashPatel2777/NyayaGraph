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

def parse_text(file_path: str):
    """
    Parses plain text files.
    """
    with logfire.span("📄 Text Parsing", filename=file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logfire.error(f"❌ Text Parse Failed: {e}")
            raise e
