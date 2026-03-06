from __future__ import annotations

import json
import logging
import os
import re
import socket
import time
from collections import OrderedDict
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from google import genai
from google.genai import types
import pdfplumber
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from pydantic import BaseModel, Field, ValidationError, model_validator
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import RequestEntityTooLarge


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

    @app.errorhandler(RequestEntityTooLarge)
    def handle_request_too_large(_: RequestEntityTooLarge) -> Any:
        return (
            jsonify(
                {
                    "error": "File too large. Please upload a smaller PDF (max 16 MB) or use a shorter report.",
                }
            ),
            413,
        )

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.post("/upload-report")
    def upload_report() -> Any:
        """Upload a PDF once and cache its extracted text for later chat turns.

        Mobile connections are often unstable for large uploads, so this endpoint
        lets the frontend upload the PDF a single time and then reference it via
        a lightweight report_id on each subsequent /chat call.
        """
        file_storage = request.files.get("file")
        if not file_storage:
            return jsonify({"error": "PDF file is required."}), 400

        try:
            saved_path = _save_pdf_upload(file_storage=file_storage, uploads_dir=uploads_dir)
            report_text = extract_pdf_text(path=saved_path)
        except (ValueError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400
        finally:
            # Best-effort cleanup
            try:
                if "saved_path" in locals():
                    saved_path.unlink(missing_ok=True)  # type: ignore[union-attr]
            except OSError:
                LOG.warning("Failed to delete temp upload in /upload-report.", exc_info=True)

        report_id, _ = _cache_report(report_text=report_text)
        return jsonify({"report_id": report_id})

    @app.post("/ingest-report-url")
    def ingest_report_url() -> Any:
        """Ingest a remote PDF URL, extract text, and cache it as report_id."""
        pdf_url = _get_pdf_url_from_request()
        if not pdf_url:
            return jsonify({"error": "pdf_url is required."}), 400

        try:
            _validate_remote_pdf_url(pdf_url)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        start = time.monotonic()
        try:
            saved_path = _download_pdf_to_temp(pdf_url=pdf_url, uploads_dir=uploads_dir)
            report_text = extract_pdf_text(path=saved_path)
        except (ValueError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400
        finally:
            try:
                if "saved_path" in locals():
                    saved_path.unlink(missing_ok=True)  # type: ignore[union-attr]
            except OSError:
                LOG.warning("Failed to delete temp download in /ingest-report-url.", exc_info=True)

        report_id, _ = _cache_report(report_text=report_text)
        LOG.info("Ingested pdf_url in %.0fms (report_id=%s).", (time.monotonic() - start) * 1000, report_id)
        return jsonify({"report_id": report_id})

    @app.post("/chat")
    def chat() -> Any:
        try:
            # Parse conversation history from JSON if provided
            history_json = request.form.get("history", "[]")
            try:
                history: List[Dict[str, str]] = json.loads(history_json)
            except (json.JSONDecodeError, TypeError):
                history = []

            report_id_raw = (request.form.get("report_id") or "").strip()
            has_report = bool(report_id_raw)

            form = ChatForm.model_validate(
                {
                    "message": (request.form.get("message") or "").strip(),
                    "has_file": bool(request.files.get("file")),
                    "has_report": has_report,
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

        report_id: Optional[str] = report_id_raw if has_report else None

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
        elif report_id:
            cached_report = REPORT_CACHE.get(report_id)
            if not cached_report:
                return (
                    jsonify(
                        {
                            "error": "This report is no longer available. Please upload the PDF again.",
                        }
                    ),
                    400,
                )
            report_text = cached_report.text
            patient_name_hint = cached_report.patient_name_hint

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

    @app.post("/translate")
    def translate() -> Any:
        try:
            body = request.get_json(silent=True) or {}
            form = TranslateForm.model_validate(
                {
                    "source_markdown": str(body.get("source_markdown") or ""),
                    "target_language": str(body.get("target_language") or "").strip(),
                }
            )
        except ValidationError as exc:
            return (
                jsonify(
                    {
                        "error": "Invalid translation request.",
                        "details": exc.errors(),
                    }
                ),
                400,
            )

        try:
            translated = translate_markdown(
                source_markdown=form.source_markdown,
                target_language=form.target_language,
            )
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500

        return jsonify({"translated_markdown": translated})

    return app


class ChatForm(BaseModel):
    message: str = Field(default="", max_length=4000)
    has_file: bool
    has_report: bool = False

    @model_validator(mode="after")
    def ensure_message_or_file(self) -> "ChatForm":
        """Ensure that at least one of message or file is provided."""
        if not self.message.strip() and not (self.has_file or self.has_report):
            raise ValueError("Either a message or a PDF file is required.")
        return self


ALLOWED_EXTENSIONS = {".pdf"}

MAX_PDF_PAGES: int = 7
MAX_REPORT_CHARS: int = 20_000
MAX_HISTORY_MESSAGES: int = 16
REPORT_CACHE_MAX_ITEMS: int = 100
REMOTE_PDF_MAX_BYTES: int = 16 * 1024 * 1024  # keep aligned with MAX_CONTENT_LENGTH


class TranslateForm(BaseModel):
    source_markdown: str = Field(min_length=1, max_length=20_000)
    target_language: str = Field(pattern="^(hi|mr)$")


def _get_pdf_url_from_request() -> str:
    if request.is_json:
        body = request.get_json(silent=True) or {}
        return str(body.get("pdf_url") or "").strip()
    return str(request.form.get("pdf_url") or "").strip()


def _allowed_remote_pdf_hosts() -> List[str]:
    raw = os.getenv("REMOTE_PDF_HOST_ALLOWLIST", "").strip()
    if raw:
        return [h.strip().lower() for h in raw.split(",") if h.strip()]
    return ["143.110.185.63", "healthnovoindia.com", ".healthnovoindia.com"]


def _is_host_allowed(host: str) -> bool:
    host_l = host.lower().strip(".")
    for allowed in _allowed_remote_pdf_hosts():
        a = allowed.strip()
        if not a:
            continue
        if a.startswith("."):
            suffix = a.lstrip(".")
            if host_l == suffix or host_l.endswith("." + suffix):
                return True
        elif host_l == a:
            return True
    return False


def _validate_resolved_ips(host: str) -> None:
    """Resolve host and block private/loopback/link-local/reserved ranges (SSRF protection)."""
    try:
        addr_infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError("Could not resolve PDF host.") from exc

    if not addr_infos:
        raise ValueError("Could not resolve PDF host.")

    for info in addr_infos:
        ip_str = info[4][0]
        ip = ip_address(ip_str)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError("PDF host resolves to a disallowed network address.")


def _validate_remote_pdf_url(pdf_url: str) -> None:
    parsed = urlparse(pdf_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("pdf_url must start with http:// or https://")
    if not parsed.netloc:
        raise ValueError("pdf_url is invalid.")
    host = parsed.hostname or ""
    if not host:
        raise ValueError("pdf_url host is invalid.")
    if host.lower() in {"localhost"}:
        raise ValueError("pdf_url host is not allowed.")
    if not _is_host_allowed(host):
        raise ValueError("pdf_url host is not allowlisted.")
    _validate_resolved_ips(host)


def _download_pdf_to_temp(pdf_url: str, uploads_dir: Path) -> Path:
    """Download a remote PDF with a hard byte cap and basic signature validation."""
    req = Request(
        pdf_url,
        headers={
            "User-Agent": "health-report-assistant/1.0",
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=20) as resp:  # nosec - URL is validated + allowlisted
            content_length = resp.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > REMOTE_PDF_MAX_BYTES:
                        raise ValueError("Remote PDF is too large.")
                except ValueError:
                    # Ignore malformed Content-Length; enforce via streaming cap below.
                    pass

            safe_name = f"{uuid4().hex}.pdf"
            dest_path = uploads_dir / safe_name
            bytes_read = 0
            with dest_path.open("wb") as f:
                first_chunk = resp.read(5)
                if first_chunk != b"%PDF-":
                    raise ValueError("Remote file does not look like a valid PDF.")
                f.write(first_chunk)
                bytes_read += len(first_chunk)

                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    bytes_read += len(chunk)
                    if bytes_read > REMOTE_PDF_MAX_BYTES:
                        raise ValueError("Remote PDF is too large.")
                    f.write(chunk)
            return dest_path
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("Failed to download the remote PDF.") from exc


class CachedReport(BaseModel):
    text: str
    patient_name_hint: Optional[str] = None


REPORT_CACHE: "OrderedDict[str, CachedReport]" = OrderedDict()


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
            total_pages = len(pdf.pages)
            for page in pdf.pages[:MAX_PDF_PAGES]:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages_text.append(page_text)
            if total_pages > MAX_PDF_PAGES:
                LOG.info(
                    "PDF truncated from %d to %d pages for processing.",
                    total_pages,
                    MAX_PDF_PAGES,
                )
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


def _cache_report(report_text: str) -> Tuple[str, Optional[str]]:
    """Store extracted report text in a small in-memory cache.

    This keeps the PDF upload as a one-time operation (especially helpful for
    mobile users), and later chat turns reference the cached text via a
    lightweight report_id instead of re-uploading the entire file.
    """
    patient_name_hint = extract_patient_name_hint(report_text)
    report_id = uuid4().hex

    REPORT_CACHE[report_id] = CachedReport(text=report_text, patient_name_hint=patient_name_hint)
    if len(REPORT_CACHE) > REPORT_CACHE_MAX_ITEMS:
        oldest_key, _ = REPORT_CACHE.popitem(last=False)
        LOG.info("Evicted oldest cached report_id: %s", oldest_key)

    LOG.info("Cached report_id=%s (length=%d).", report_id, len(report_text))
    return report_id, patient_name_hint


SYSTEM_INSTRUCTION = """Role: Highly intelligent medical administrative assistant.

Constraints:
- You are NOT a doctor.
- Do NOT provide diagnoses.
- Do NOT provide treatment advice.

Core job (when a PDF is provided):
- When HEALTH_REPORT_TEXT contains actual report content, identify abnormal biomarkers strictly using the reference ranges in the report.
- Recommend the single best type of medical specialist (e.g., Hematologist, Cardiologist) for the user to consult.

Patient name vs. tester name:
- Use ONLY the patient's name (e.g. "Patient Name", "Name of Patient", "Patient:"). Do NOT use tester name, technician name, "Collected by", "Reported by", or any other staff name as the patient identity. If the report has no clear patient name field, use "Patient: Unknown"—never substitute with a tester or lab staff name.

Report value accuracy:
- Stick strictly to the values as printed in the report. Do not approximate, round differently, or invent values. Report the exact biomarker name, numeric value, and reference range from the report.
- Identify BOTH above-reference (high) AND below-reference (low) abnormal values. Many reports include tests that are low—capture them. Each abnormal finding in key_findings MUST specify the direction: "high" or "low" based on the reference range.

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
- When you are summarizing a specific patient's lab results from an actual HEALTH_REPORT_TEXT, start the Markdown with the patient's name from the report if available (patient name ONLY, never tester/staff names). If no patient name is identifiable in the report, use "Patient: Unknown".

Output requirements:
- Return ONLY a valid JSON object.
- The JSON object MUST contain these keys exactly: doctor_type, reasoning, urgency_level, key_findings, reply_markdown
- reply_markdown MUST be Markdown. For lab-result summaries, it SHOULD start with the patient's name (patient name field only—never tester or staff name) or \"Patient: Unknown\" if the name is not present. For pure chit-chat or UX questions with no PDF, a normal conversational opening is fine and MUST NOT claim you analyzed a report.
- doctor_type MUST be a single specialist type (string).
- urgency_level MUST be one of: low, medium, high
- key_findings MUST be an array of concise strings listing abnormal biomarkers and the direction (high or low) based on the provided reference ranges when a report is present. Include BOTH high and low abnormals. Use the exact values from the report. Empty array when no report is provided."""


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

    # Optionally truncate very large reports to keep latency low and prompts manageable.
    if report_text and len(report_text) > MAX_REPORT_CHARS:
        LOG.info(
            "Report text truncated from %d to %d characters before sending to Gemini.",
            len(report_text),
            MAX_REPORT_CHARS,
        )
        report_text = report_text[:MAX_REPORT_CHARS]

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

    # Update conversation history (truncate to the most recent messages to keep prompts bounded).
    updated_history = (conversation_history or []) + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": payload["reply_markdown"]},
    ]
    if len(updated_history) > MAX_HISTORY_MESSAGES:
        updated_history = updated_history[-MAX_HISTORY_MESSAGES:]

    return payload, updated_history


def translate_markdown(source_markdown: str, target_language: str) -> str:
    """Translate an existing Markdown report into Hindi or Marathi.

    The translation must preserve ALL numeric values, units, biomarker
    names, and reference ranges exactly as in the source. Only natural
    language around those values may change.
    """
    api_key = _get_env("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    model_name = _get_env("GEMINI_MODEL", default="gemini-3-flash-preview")

    if target_language == "hi":
        language_label = "Hindi"
    elif target_language == "mr":
        language_label = "Marathi"
    else:
        raise RuntimeError("Unsupported target language.")

    system_instruction = (
        "You are a precise medical report translator. Translate the following "
        f"Markdown into {language_label}. You MUST:\n"
        "- Preserve all biomarker names exactly as written.\n"
        "- Preserve all numeric values, units, and reference ranges exactly as written.\n"
        "- Do not add or remove any findings, biomarkers, or advice.\n"
        "- Keep the same overall structure (headings, bullet points) where possible.\n"
        "- Output ONLY the translated Markdown, no explanations.\n"
    )

    prompt = f"{system_instruction}\n\n---\n\nSOURCE MARKDOWN:\n{source_markdown}"

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=4096,
            ),
        )
    except Exception as exc:
        LOG.exception("Gemini translation request failed.", exc_info=True)
        raise RuntimeError(f"Translation failed: {exc}") from exc

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Translation failed: empty response from model.")
    return text


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
    app.run(host="0.0.0.0", port=8000, debug=True)
