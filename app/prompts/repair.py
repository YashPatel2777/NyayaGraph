"""One-shot "fix this JSON" retry (R-22, E-02).

Sent as a user message after the model's invalid reply, which stays in the conversation as
the assistant message. Placeholder: {validation_error}.
"""

REPAIR_PROMPT = """Your previous reply could not be used: it was not valid JSON or it did not match the required schema.

Validation error:
{validation_error}

Return the corrected reply as a single JSON object that matches the required schema. Keep every value that was already valid. Do not add commentary, markdown or code fences."""
