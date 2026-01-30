# Changelog

## 2026-01-28

- Added Flask backend (`main.py`) with:
  - Secure PDF upload handling (type/extension checks, size limit, temp storage in instance folder)
  - PDF text extraction via `pdfplumber`
  - Gemini integration via `google-generativeai` enforcing medical-admin constraints and structured keys
  - `/chat` endpoint returning JSON (`doctor_type`, `reasoning`, `urgency_level`, `key_findings`, `reply_markdown`)
- Added frontend chat UI (`templates/index.html`) with:
  - Async `fetch()` chat calls (no page reload)
  - PDF upload input
  - Markdown rendering with sanitization
- Added `requirements.txt`
- Fixed `.env` loading to reliably read `GEMINI_API_KEY` from the project folder (next to `main.py`), even when running from another working directory.
- Updated Gemini instructions to allow friendly chit-chat while keeping medical constraints, and added explicit app UI context so the model answers “how to upload PDF” correctly.

