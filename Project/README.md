# Document Extraction Pipeline

A FastAPI service that extracts structured data from documents (PDF, scanned images, plain text, email threads and JSON exports) using LLMs on Groq. Every file type is returned in the same JSON format, with validation, a confidence level and data-quality flags.

---

## Quick Start

### Prerequisites

- Python 3.10 or newer
- Git
- A free [Groq API key](https://console.groq.com/keys)
- *(Optional)* Tesseract OCR, used as an extra hint when reading images

### 1. Clone the repository

pull the code from github

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows (PowerShell or Command Prompt)**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

You should now see `(venv)` at the start of your terminal line.

### 4. Install the dependencies

```bash
pip install -r requirements.txt
```

### 5. Add your API key

Copy the example file and put your Groq key in it.

**Windows**
```bash
copy .env.example .env
```

**macOS / Linux**
```bash
cp .env.example .env
```

Then open `.env` and set:

```
GROQ_API_KEY=your_key_here
```

### 6. Run the server

```bash
uvicorn main:app --reload
```

The server starts at **http://127.0.0.1:8000**.

### 7. Try it

Open the interactive API page in your browser:

**http://127.0.0.1:8000/docs**

Click `POST /process-document`, then **Try it out**, choose a file and press **Execute**.

Or use the command line:

```bash
curl -X POST "http://127.0.0.1:8000/process-document" -F "file=@your_document.pdf"
```

---

## Run Without the Server (Batch Mode)

To process several files at once and save the results to `output.json`:

```bash
python main.py doc1_service_contract.pdf doc2_scanned_invoice.png doc3_project_summary.txt doc4_email_thread.txt doc5_logistics_data_export.json
```

---

## Supported Files

| Type | Extension | How it is read |
|---|---|---|
| PDF (text-based) | `.pdf` | PDF loader |
| Scanned image | `.png`, `.jpg`, `.jpeg` | OCR hint, then a vision model transcribes it |
| Project summary | `.txt` | Text loader |
| Email thread | `.txt` | Text loader |
| JSON data export | `.json` | JSON loader with code-level checks |

A `.txt` file is treated as an email thread if its filename contains "email" or "mail", or if it starts with `From:` / `Subject:` headers. Otherwise it is treated as a summary.

---

## Output Format

```json
{
  "filename": "doc1_service_contract.pdf",
  "status": "success",
  "result": {
    "document_type": "contract",
    "summary": "…",
    "confidence": "HIGH",
    "confidence_reasons": ["…"],
    "data_quality_issues": ["…"],
    "parties_involved": ["…"],
    "key_dates": ["2024-11-14"],
    "amounts": [
      { "value": 285000, "currency": "USD", "description": "Total contract value" }
    ],
    "key_identifiers": { "contract_reference": "ZT-MSA-2024-0047" },
    "raw_text": null
  }
}
```

`raw_text` is filled only for images. It contains the exact text read from the image.

---

## Architecture

```
                    ┌─────────────────────┐
                    │   Upload Document   │
                    └──────────┬──────────┘
                               ▼
                  ┌─────────────────────────┐
                  │ identify_document_type()│
                  └────────────┬────────────┘
        ┌──────────────┬───────┴───────┬──────────────┐
        ▼              ▼               ▼              ▼
       PDF           Image        TXT (email /      JSON
        │              │           summary)           │
   PDF loader    OCR → Vision LLM  Text loader   JSON loader
        │         (transcription)      │          + code checks
        └──────────────┴───────┬───────┴──────────────┘
                               ▼
                    LangChain prompt → LLM (Groq)
                               ▼
                    Pydantic schema (parse + clean)
                               ▼
                 Validation · Confidence · Quality issues
                               ▼
                        Structured JSON
```

**What each stage does**

1. **Identify the type.** `identify_document_type()` picks a flow from the file extension and filename.
2. **Load the content.** Each flow has its own loader. Images are first read by OCR, then a vision model transcribes them. JSON files also get deterministic checks in code (duplicates, missing fields, mixed currencies, inconsistent date formats).
3. **Extract with an LLM.** A shared set of rules plus a short note per document type goes to the model. The model returns JSON.
4. **Parse with Pydantic.** The `ExtractedDocument` schema checks the structure and cleans dates, amounts and identifiers.
5. **Validate.** The code checks that dates are well formed, that every extracted identifier and amount appears in the source text, and that totals add up. Problems are added to `data_quality_issues` and the confidence is lowered.

---

## Configuration

Optional settings in `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | none (required) | Your Groq API key |
| `TEXT_MODEL` | `openai/gpt-oss-120b` | Model for extraction |
| `VISION_MODEL` | `qwen/qwen3.8-27b` | Model for reading images |
| `DEBUG` | off | Set to `1` to show real error messages in API responses (development only) |

---

## Project Structure

```
├── main.py             # the whole pipeline (API + extraction + validation)
├── requirements.txt    # Python dependencies
├── .env.example        # template for your API key
└── README.md
```

The `uploads/` folder is created automatically while the server runs. Uploaded files are deleted after processing.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `venv\Scripts\activate` is blocked on Windows | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. |
| `uvicorn` is not recognized | The environment is not active. Activate it (step 3) and try again. |
| `GROQ_API_KEY` error on start | Check that `.env` exists in the project folder and contains your key. |
| `429` / rate limit errors | The app waits and retries automatically. For many files, run them one at a time. |
| "Tesseract not available" warning | This is safe to ignore. Images are still read by the vision model. To enable OCR, install [Tesseract](https://github.com/tesseract-ocr/tesseract). |
| Port 8000 already in use | Run `uvicorn main:app --reload --port 8001`. |
| A scanned PDF returns an error | The PDF has no text layer. Convert its pages to images and upload those. |
| Any other 500 error | Set `DEBUG=1` in `.env`, restart, and read the error message. |