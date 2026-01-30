from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from google import genai
from google.genai import types
import pdfplumber
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from pydantic import BaseModel, Field, ValidationError, model_validator
from werkzeug.datastructures import FileStorage


# Always load .env from the project directory (next to this file).
# override=True so the project .env wins over any existing shell/system env.
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)


LOG = logging.getLogger("health_report_assistant")


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="templates")

    # Security: limit upload size (bytes). Adjust if your reports are larger.
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MiB

    # Use instance folder for temporary uploads (not served publicly).
    uploads_dir = Path(app.instance_path) / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    _configure_logging()

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.post("/chat")
    def chat() -> Any:
        try:
            # Parse conversation history from JSON if provided
            history_json = request.form.get("history", "[]")
            try:
                history: List[Dict[str, str]] = json.loads(history_json)
            except (json.JSONDecodeError, TypeError):
                history = []

            form = ChatForm.model_validate(
                {
                    "message": (request.form.get("message") or "").strip(),
                    "has_file": bool(request.files.get("file")),
                }
            )
        except ValidationError as exc:
            return (
                jsonify(
                    {
                        "error": "Invalid request.",
                        "details": exc.errors(),
                    }
                ),
                400,
            )

        file_storage = request.files.get("file")
        report_text = ""
        patient_name_hint: Optional[str] = None

        if file_storage:
            try:
                saved_path = _save_pdf_upload(file_storage=file_storage, uploads_dir=uploads_dir)
                report_text = extract_pdf_text(path=saved_path)
                patient_name_hint = extract_patient_name_hint(report_text)
            except (ValueError, OSError) as exc:
                return jsonify({"error": str(exc)}), 400
            finally:
                # Best-effort cleanup
                try:
                    if "saved_path" in locals():
                        saved_path.unlink(missing_ok=True)  # type: ignore[union-attr]
                except OSError:
                    LOG.warning("Failed to delete temp upload.", exc_info=True)

        user_message = form.message
        if not user_message.strip() and report_text:
            # Auto-analyze when only a PDF is provided.
            user_message = (
                "Please analyze this health report, identify all abnormal biomarkers based on the "
                "provided reference ranges, and recommend the single best specialist type to consult."
            )

        try:
            gemini_payload, updated_history = call_gemini(
                user_message=user_message,
                report_text=report_text,
                patient_name_hint=patient_name_hint,
                conversation_history=history,
            )
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500

        return jsonify({**gemini_payload, "history": updated_history})

    return app


class ChatForm(BaseModel):
    message: str = Field(default="", max_length=4000)
    has_file: bool

    @model_validator(mode="after")
    def ensure_message_or_file(self) -> "ChatForm":
        """Ensure that at least one of message or file is provided."""
        if not self.message.strip() and not self.has_file:
            raise ValueError("Either a message or a PDF file is required.")
        return self


ALLOWED_EXTENSIONS = {".pdf"}


def _save_pdf_upload(file_storage: FileStorage, uploads_dir: Path) -> Path:
    """Save an uploaded PDF securely to a temporary location.

    Args:
        file_storage: Incoming file from Flask/Werkzeug.
        uploads_dir: Directory to store temporary uploads.

    Returns:
        Path to the saved PDF.

    Raises:
        ValueError: If the file is missing/invalid/not a PDF.
        OSError: If saving fails.
    """
    original_name = (file_storage.filename or "").strip()
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Only PDF files are supported.")

    # Basic content sniff: ensure it looks like a PDF.
    head = file_storage.stream.read(5)
    file_storage.stream.seek(0)
    if head != b"%PDF-":
        raise ValueError("Uploaded file does not look like a valid PDF.")

    safe_name = f"{uuid4().hex}.pdf"
    dest_path = uploads_dir / safe_name
    file_storage.save(dest_path)
    return dest_path


def extract_pdf_text(path: Path) -> str:
    """Extract text content from a PDF using pdfplumber.

    Args:
        path: Path to a PDF file.

    Returns:
        Extracted text (may be large).

    Raises:
        ValueError: If extraction fails or yields empty text.
    """
    try:
        with pdfplumber.open(str(path)) as pdf:
            pages_text: List[str] = []
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages_text.append(page_text)
    except (OSError, ValueError) as exc:
        raise ValueError("Could not read the PDF. Please upload a text-based report PDF.") from exc

    text = "\n\n".join(pages_text).strip()
    if not text:
        raise ValueError("No extractable text found in the PDF (it may be a scanned image).")
    return text


_PATIENT_NAME_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(?im)^\s*patient\s*name\s*[:\-]\s*(?P<name>[A-Z][A-Za-z .'-]{1,80})\s*$"),
    re.compile(r"(?im)^\s*name\s*[:\-]\s*(?P<name>[A-Z][A-Za-z .'-]{1,80})\s*$"),
)


def extract_patient_name_hint(report_text: str) -> Optional[str]:
    """Heuristically extract a patient name from report text (best-effort).

    This is used only as a hint to the model. If uncertain, returns None.
    """
    for pattern in _PATIENT_NAME_PATTERNS:
        match = pattern.search(report_text)
        if match:
            name = (match.group("name") or "").strip()
            if 2 <= len(name) <= 80:
                return name
    return None


SYSTEM_INSTRUCTION = """Role: Highly intelligent medical administrative assistant.

Constraints:
- You are NOT a doctor.
- Do NOT provide diagnoses.
- Do NOT provide treatment advice.

Core job (when a PDF is provided):
- When HEALTH_REPORT_TEXT contains actual report content, identify abnormal biomarkers strictly using the reference ranges in the report.
- Recommend the single best type of medical specialist (e.g., Hematologist, Cardiologist) for the user to consult.

Behavior when NO PDF is provided (HEALTH_REPORT_TEXT is \"(No PDF provided.)\"):
- Do NOT invent or assume any patient name.
- Do NOT invent or assume any biomarker values or lab results.
- Do NOT pretend to have analyzed a report.
- You MAY do friendly chit-chat (e.g., greetings, \"How are you doing today?\").
- You SHOULD explain briefly how to upload the PDF and what you can do once you have it.
- In this case, key_findings MUST be an empty array, and doctor_type SHOULD be a generic value like \"Not determined\" or \"General Physician\".

Conversation scope (general):
- You MAY do friendly chit-chat (e.g., greetings, short follow-ups).
- You SHOULD answer questions about how to use this web app (e.g., how to upload the PDF).

Formatting:
- Use Markdown for clarity.
- When you are summarizing a specific patient's lab results from an actual HEALTH_REPORT_TEXT, start the Markdown with the patient's name from the report if available.

Output requirements:
- Return ONLY a valid JSON object.
- The JSON object MUST contain these keys exactly: doctor_type, reasoning, urgency_level, key_findings, reply_markdown
- reply_markdown MUST be Markdown. For lab-result summaries, it SHOULD start with the patient's name (or \"Patient: Unknown\" if the name is not present derive it from file name or check the file twice and answer on basis of report after that chat ). For pure chit-chat or UX questions with no PDF, a normal conversational opening is fine and MUST NOT claim you analyzed a report.
- doctor_type MUST be a single specialist type (string).
- urgency_level MUST be one of: low, medium, high
- key_findings MUST be an array of concise strings listing abnormal biomarkers and the direction (high/low) based on the provided reference ranges when a report is present, or an empty array when no report is provided."""


APP_UI_CONTEXT = """App UI context:
- There is a 'PDF:' file picker next to the message box.
- To upload: choose a PDF using that file picker, then click 'Send'.
- The file is optional; without a PDF you can still ask general workflow questions."""


def call_gemini(
    user_message: str,
    report_text: str,
    patient_name_hint: Optional[str],
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Call Gemini and return a validated JSON payload for the frontend.

    Args:
        user_message: The user's chat message.
        report_text: Extracted text from the PDF report (can be empty).
        patient_name_hint: Best-effort extracted name to help the model.
        conversation_history: Previous messages in format [{"role": "user", "content": "..."}, ...].

    Returns:
        Tuple of (payload dict, updated_history list).

    Raises:
        RuntimeError: If Gemini is not configured or responses are unusable.
    """
    api_key = _get_env("GEMINI_API_KEY")
    # New google-genai client uses the stable v1 Gemini API.
    client = genai.Client(api_key=api_key)
    model_name = _get_env("GEMINI_MODEL", default="gemini-3-flash-preview")

    # Build context + (optional) conversation history into a single prompt.
    context_parts: List[str] = [APP_UI_CONTEXT]
    if report_text.strip():
        context_parts.append("HEALTH_REPORT_TEXT:\n" + report_text)
    else:
        context_parts.append("HEALTH_REPORT_TEXT:\n(No PDF provided.)")

    if patient_name_hint:
        context_parts.append(f"PATIENT_NAME_HINT:\n{patient_name_hint}")

    # Include prior turns as a textual transcript for context.
    conversation_lines: List[str] = []
    if conversation_history:
        for msg in conversation_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not content:
                continue
            if role == "user":
                conversation_lines.append(f"USER: {content}")
            elif role == "assistant":
                conversation_lines.append(f"ASSISTANT: {content}")

    if conversation_lines:
        context_parts.append("CONVERSATION_HISTORY:\n" + "\n".join(conversation_lines))

    context_parts.append("USER_MESSAGE:\n" + user_message)
    full_prompt = "\n\n---\n\n".join(context_parts)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=4096,
            ),
        )
    except Exception as exc:  # google sdk raises varied exceptions
        LOG.exception("Gemini request failed.", exc_info=True)
        raise RuntimeError(f"Gemini request failed: {exc}") from exc

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response.")

    parsed = _parse_json_best_effort(text)
    if parsed is None:
        # Second pass: ask Gemini to reformat into strict JSON only.
        repair = _repair_to_strict_json(client=client, model_name=model_name, raw=text)
        parsed = _parse_json_best_effort(repair)

    if parsed is None:
        # Fallback: treat the whole model text as Markdown reply and
        # synthesize minimal structured fields instead of returning 500.
        payload = _normalize_payload(
            {
                "doctor_type": "General Physician",
                "reasoning": "Model returned a non-JSON answer; using it as fallback Markdown.",
                "urgency_level": "medium",
                "key_findings": [],
                "reply_markdown": text,
            }
        )
    else:
        payload = _normalize_payload(parsed)

    # Update conversation history
    updated_history = (conversation_history or []) + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": payload["reply_markdown"]},
    ]

    return payload, updated_history


def _parse_json_best_effort(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if fenced:
        try:
            obj2 = json.loads(fenced.group(1))
            if isinstance(obj2, dict):
                return obj2
        except json.JSONDecodeError:
            return None
    return None


def _repair_to_strict_json(client: genai.Client, model_name: str, raw: str) -> str:
    repair_instruction = (
        "Convert the following content into a STRICT JSON object ONLY. "
        "No markdown fences, no extra text. "
        "The JSON MUST contain: doctor_type, reasoning, urgency_level, key_findings, reply_markdown.\n\n"
        f"CONTENT:\n{raw}"
    )
    try:
        r = client.models.generate_content(
            model=model_name,
            contents=repair_instruction,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=2048,
            ),
        )
        return (getattr(r, "text", None) or "").strip()
    except Exception:
        LOG.warning("Gemini repair call failed.", exc_info=True)
        return ""


def _normalize_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    doctor_type = str(obj.get("doctor_type") or "").strip() or "General Physician"
    reasoning = str(obj.get("reasoning") or "").strip()
    urgency_level = str(obj.get("urgency_level") or "").strip().lower()
    key_findings_raw = obj.get("key_findings")
    reply_markdown = str(obj.get("reply_markdown") or "").strip()

    if urgency_level not in {"low", "medium", "high"}:
        urgency_level = "medium"

    key_findings: List[str] = []
    if isinstance(key_findings_raw, list):
        for item in key_findings_raw:
            if isinstance(item, str) and item.strip():
                key_findings.append(item.strip())

    if not reply_markdown:
        # Ensure frontend always has something to display.
        reply_markdown = (
            "Patient: Unknown\n\n"
            "I couldn’t generate a formatted summary, but I can still help identify abnormal biomarkers "
            "if you upload the PDF report."
        )

    return {
        "doctor_type": doctor_type,
        "reasoning": reasoning,
        "urgency_level": urgency_level,
        "key_findings": key_findings,
        "reply_markdown": reply_markdown,
    }


def _get_env(key: str, default: Optional[str] = None) -> str:
    env_val = os.getenv(key)
    if env_val is not None and env_val.strip():
        return env_val.strip()
    if default is not None:
        return default
    raise RuntimeError(f"Missing required environment variable: {key}")


def _mask_key(key: str) -> str:
    """Return a safe mask for logging (e.g. ...abcd)."""
    if not key or len(key) < 4:
        return "****"
    return "..." + key.strip()[-4:]


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


app = create_app()

# Log which API key is active at startup (masked) so you can confirm .env is used.
try:
    _key = _get_env("GEMINI_API_KEY")
    LOG.info("GEMINI_API_KEY loaded: %s", _mask_key(_key))
except RuntimeError:
    pass


if __name__ == "__main__":
    # For local dev only. Use a production WSGI server in deployment.
    app.run(host="0.0.0.0", port=5000, debug=True)
