# ActumAI Extract

**Intelligent Document Processing & Structured Extraction Engine**

ActumAI Extract converts unstructured business documents — multi-page
commercial invoices, legal contracts, financial reports — into rigorously
validated, deeply nested JSON, using an LLM constrained by `instructor` and
`Pydantic v2`. It is built as a reference-quality example of production
LLMOps: type-safe configuration, structured JSON logging with correlation
IDs, a self-healing extraction loop, and a clean separation between
parsing, extraction, and API layers.

---

## 1. Architecture

```
                                   ┌────────────────────────────┐
                                   │        Client / API         │
                                   │  (curl, Postman, ERP, etc.) │
                                   └──────────────┬───────────────┘
                                                  │ HTTPS
                                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              FastAPI (main.py)                            │
│                                                                            │
│   RequestContextMiddleware ── injects X-Request-ID, logs timing           │
│                                                                            │
│   ┌────────────────────┐   ┌────────────────────┐   ┌─────────────────┐  │
│   │ POST /extract/text │   │ POST /extract/pdf   │   │   GET /health   │  │
│   └──────────┬──────────┘   └──────────┬──────────┘   └─────────────────┘  │
│              │                          │                                 │
│              │                          ▼                                 │
│              │              ┌────────────────────────┐                   │
│              │              │  DocumentParserService  │                   │
│              │              │  (services/document_    │                   │
│              │              │   parser.py — pypdf)     │                   │
│              │              └───────────┬──────────────┘                   │
│              │                          │ clean plain text                │
│              ▼                          ▼                                 │
│         ┌─────────────────────────────────────────┐                      │
│         │           ExtractionEngine                │                      │
│         │        (extractors/engine.py)              │                      │
│         │                                             │                      │
│         │   instructor(AsyncOpenAI) ── tool-calling    │                      │
│         │   tenacity ── upstream resilience (retry/     │                      │
│         │                backoff on 429 / timeout)       │                      │
│         │                                             │                      │
│         │   ┌─────────────────────────────────────┐   │                      │
│         │   │       SELF-HEALING LOOP               │   │                      │
│         │   │  1. call LLM with response_model      │   │                      │
│         │   │  2. Pydantic validates output          │   │                      │
│         │   │  3. on ValidationError:                │   │                      │
│         │   │       - format field-level feedback    │   │                      │
│         │   │       - append to conversation          │   │                      │
│         │   │       - retry (bounded)                │   │                      │
│         │   │  4. on success: return validated model │   │                      │
│         │   └─────────────────────────────────────┘   │                      │
│         └───────────────────┬─────────────────────────┘                      │
│                             │ validated InvoiceExtractionResult              │
│                             ▼                                                │
│         ┌─────────────────────────────────────────┐                        │
│         │      Global Exception Handlers            │                        │
│         │  DocumentParsingError   -> 400             │                        │
│         │  ExtractionValidationError -> 422          │                        │
│         │  UpstreamError          -> 502             │                        │
│         │  pydantic.ValidationError -> 422           │                        │
│         │  Exception (catch-all)  -> 500             │                        │
│         └─────────────────────────────────────────┘                        │
└──────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
                                   ┌────────────────────────────┐
                                   │   LLM Provider (OpenAI /     │
                                   │   Azure OpenAI / vLLM, etc.) │
                                   └────────────────────────────┘
```

### Layering principles

| Layer                    | File                            | Responsibility                                                            |
|---------------------------|----------------------------------|-----------------------------------------------------------------------------|
| Configuration             | `core/config.py`                | Single typed source of truth, `pydantic-settings`, `.env` support          |
| Observability             | `core/logger.py`                | Structured JSON logs, request-ID correlation via `contextvars`             |
| Domain schema             | `schemas/invoice.py`            | Nested Pydantic v2 models + cross-field business validators                |
| Document ingestion        | `services/document_parser.py`   | PDF → clean text, isolated from extraction concerns                        |
| Extraction orchestration  | `extractors/engine.py`          | `instructor` + `tenacity`, self-healing retry loop                         |
| HTTP surface              | `main.py`                       | Routing, middleware, dependency injection, global exception translation    |

---

## 2. The Self-Healing Extraction Loop

LLMs occasionally produce structurally plausible but numerically or
semantically invalid output — a line item where `quantity * unit_price !=
net_amount`, an invalid NIP checksum, a payment due date before the issue
date. Rather than surfacing this as an opaque 500 error, `ExtractionEngine`
treats a `pydantic.ValidationError` as **actionable signal**:

1. The LLM is called with the target `response_model` (e.g.
   `InvoiceExtractionResult`) via `instructor`'s tool-calling mode.
2. If the returned payload fails Pydantic validation, the engine captures
   the **exact** `ValidationError`, including every offending field path
   and the precise reason it failed.
3. `_format_validation_feedback` renders this into a corrective instruction
   — *"Field 'line_items.0.net_amount': Line item #1 arithmetic mismatch:
   quantity (10) * unit_price_net (100.00) = 1000.00, but net_amount was
   reported as 500.00."* — and appends it as a new user turn.
4. The model is re-invoked with the full corrective context and asked to
   re-emit the complete, corrected object.
5. This repeats up to `EXTRACTION_MAX_RETRIES` times (configurable). If the
   budget is exhausted, an `ExtractionValidationError` is raised, and the
   API layer returns a `422` with the full list of unresolved field errors
   for human review.

This loop is orthogonal to **upstream resilience**: transient provider
failures (rate limits, timeouts, connection errors) are retried
independently via a `tenacity`-managed exponential backoff-with-jitter
policy around each individual LLM call, and are never treated as
self-healable validation problems.

---

## 3. Running the service

### 3.1 Prerequisites

- Python 3.11+
- An OpenAI-compatible API key

### 3.2 Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

### 3.3 Run the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The interactive OpenAPI docs are then available at
`http://localhost:8000/docs`.

### 3.4 Run the test suite

```bash
pytest -v
```

All extraction-engine tests run fully offline via `monkeypatch` — no live
LLM calls or API key are required to run the test suite.

---

## 4. Example requests

### 4.1 Health check

```bash
curl -s http://localhost:8000/health | jq
```

### 4.2 Extract from raw text

```bash
curl -s -X POST http://localhost:8000/api/v1/extract/text \
  -H "Content-Type: application/json" \
  -d '{
    "document_text": "FAKTURA VAT nr FV/2024/09/00123\nData wystawienia: 2024-09-01\nSprzedawca: Acme Manufacturing Sp. z o.o., NIP 123-456-32-18\nNabywca: Global Imports GmbH\nLp. 1 Widget X-200, 10 szt. x 100.00 PLN netto, VAT 23%, wartość netto 1000.00, VAT 230.00, brutto 1230.00\nRazem netto: 1000.00 PLN, VAT: 230.00 PLN, brutto: 1230.00 PLN\nTermin płatności: 2024-09-15"
  }' | jq
```

### 4.3 Extract from an uploaded PDF

```bash
curl -s -X POST http://localhost:8000/api/v1/extract/pdf \
  -F "file=@/path/to/invoice.pdf;type=application/pdf" | jq
```

### 4.4 Example response envelope

```json
{
  "data": {
    "document_type": "commercial_invoice",
    "document_number": "FV/2024/09/00123",
    "currency": "PLN",
    "seller": { "legal_name": "Acme Manufacturing Sp. z o.o.", "tax_id_nip": "1234563218", "...": "..." },
    "buyer": { "legal_name": "Global Imports GmbH", "...": "..." },
    "line_items": [ { "position_number": 1, "net_amount": "1000.00", "vat_amount": "230.00", "gross_amount": "1230.00", "...": "..." } ],
    "payment": { "issue_date": "2024-09-01", "payment_due_date": "2024-09-15" },
    "total_net_amount": "1000.00",
    "total_vat_amount": "230.00",
    "total_gross_amount": "1230.00"
  },
  "metadata": {
    "attempts_made": 1,
    "self_healed": false,
    "total_latency_seconds": 2.14,
    "model_used": "gpt-4o-2024-08-06"
  }
}
```

---

## 5. Configuration reference

All settings are environment variables, loaded via `pydantic-settings`
(see `core/config.py` for the authoritative, documented list). Key
variables:

| Variable                       | Default                | Description                                             |
|---------------------------------|-------------------------|-----------------------------------------------------------|
| `OPENAI_API_KEY`                | —                        | LLM provider API key (required)                          |
| `OPENAI_BASE_URL`                | provider default         | Override for Azure OpenAI / self-hosted gateways          |
| `EXTRACTION_MODEL`               | `gpt-4o-2024-08-06`      | Model used for structured extraction                      |
| `EXTRACTION_MAX_RETRIES`         | `3`                       | Max self-healing correction attempts                       |
| `LLM_REQUEST_TIMEOUT_SECONDS`    | `60`                      | Per-request LLM HTTP timeout                                |
| `MAX_UPLOAD_SIZE_BYTES`          | `15728640` (15 MB)        | Max accepted PDF upload size                                |
| `MAX_PDF_PAGES`                  | `100`                     | Max pages processed per PDF                                 |
| `LOG_LEVEL`                      | `INFO`                    | Root structured logging level                                |

---

## 6. Error handling contract

| Exception                     | HTTP status | Meaning                                                      |
|--------------------------------|-------------|------------------------------------------------------------------|
| `DocumentParsingError`         | 400         | Corrupted/encrypted/unreadable PDF, or empty text input           |
| File too large                 | 413         | Upload exceeds `MAX_UPLOAD_SIZE_BYTES`                             |
| `pydantic.ValidationError`      | 422         | Malformed request payload                                          |
| `ExtractionValidationError`     | 422         | Self-healing budget exhausted; includes field-level error detail   |
| `UpstreamError`                 | 502         | LLM provider failure (rate limit, timeout, 5xx) after retry budget |
| Unhandled `Exception`           | 500         | Logged with full traceback + request ID; generic message to client |

Every response — success or failure — carries an `X-Request-ID` header for
end-to-end log correlation.
