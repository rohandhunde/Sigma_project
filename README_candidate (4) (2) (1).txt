════════════════════════════════════════════════════════════════
SIGMA SOLVE — AI ENGINEER ASSESSMENT
Standard Extraction Task
════════════════════════════════════════════════════════════════

OVERVIEW
────────
You have been given five documents. Your task is to build a Python
pipeline that extracts structured information from all five, validates
the extracted data, and flags any records where extraction confidence
is low or data quality is suspect.

THE FIVE DOCUMENTS
──────────────────
doc1_service_contract.pdf      PDF contract — text-based, multi-party, tables
doc2_scanned_invoice.png       Image scan — requires OCR, partial handwriting
doc3_project_summary.txt       Plain text — unstructured prose, no labels
doc4_email_thread.txt          Email thread — multi-sender, key info scattered
doc5_logistics_data_export.json JSON export — structured but noisy

WHAT YOUR PIPELINE MUST DO
───────────────────────────
For each document, extract at minimum:

  - document_type      (contract / invoice / summary / email_thread / data_export)
  - parties_involved   (organisations and/or people — as many as relevant)
  - key_dates          (normalised to YYYY-MM-DD where possible)
  - amounts            (monetary values with currency — include all significant ones)
  - key_identifiers    (contract numbers, invoice numbers, shipment IDs, PO refs etc)
  - summary            (2-3 sentence plain-English summary of what the document covers)

Additionally:

  - Define and apply a validation schema to each extracted record
  - Flag any document where confidence in the extraction is LOW and explain why
  - Flag any data quality issues you detect (missing fields, format inconsistencies,
    duplicates, conflicting data, currency mismatches — anything that would matter
    in a real production system)

WHAT WE WILL LOOK AT
─────────────────────
1. Your extraction strategy per document type — did you make deliberate choices?
2. Whether your validation layer catches the issues in the documents (they are there)
3. How you handle the documents that are harder — the image, the prose, the noise
4. Code quality — modular, readable, commented
5. Your README — explain your choices, not just what the code does

WHAT TO SUBMIT
───────────────
- Your pipeline code (single file or small repo)
- A README explaining your extraction strategy per document type and your key decisions
- The output of your pipeline run against these five documents

You do not need to build a UI or a web service. A script that runs and produces
structured output (JSON, CSV, or printed) is sufficient.

You may use any Python libraries, LLM APIs, or AI services you choose.
If you use an LLM for extraction, document your prompting strategy.
If you use OCR, document which library and why.

TIME
────
This task is designed to take 2-3 hours. We are not looking for a production system.
We are looking for how you think about extraction problems and what you do when
documents are ambiguous or incomplete.

SUBMISSION
──────────
Submit your code and README to the hiring contact within 48 hours of receiving
this brief.

════════════════════════════════════════════════════════════════
Good luck.
════════════════════════════════════════════════════════════════
