# Changelog

## 2026-02-11

- Updated `SYSTEM_INSTRUCTION` prompt:
  - **Patient name only:** Explicit rule to use only the patient's name field—never tester, technician, "Collected by", or staff names. Use "Patient: Unknown" if no patient name is in the report.
  - **Report value accuracy:** Instruct model to stick to exact values from the report, include both high and low abnormal biomarkers, and specify direction (high/low) for each finding.
- Added PDF performance + mobile reliability improvements:
  - Backend: capped PDF page extraction + report text length before Gemini.
  - Frontend: upload PDF once and reuse `report_id` across chat turns (prevents repeated large uploads).
- Added deep-link support for report PDFs:
  - Backend: `POST /ingest-report-url` ingests a remote `pdf_url` with allowlisting + SSRF protections and returns `report_id`.
  - Frontend: reads `?pdfurl=...`, ingests automatically, clears the query params, and auto-runs analysis.
- Improved error handling:
  - Returns JSON for 413 “file too large” so the UI can show a clear message.
- Added multilingual report view:
  - Backend: `/translate` endpoint translates existing Markdown summaries into Hindi or Marathi while preserving all biomarker names, numbers, units, and reference ranges.
  - Frontend: adds “View this report in: Hindi / Marathi” controls under each assistant report bubble, calling `/translate` and rendering the translated Markdown.

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

