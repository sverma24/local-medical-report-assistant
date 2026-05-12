# Local Medical Report Assistant

Streamlit app for private, local-first medical report understanding using:

- `Ollama` + `gemma4:26b` for reasoning
- `ChromaDB` for local vector retrieval
- `LangChain` + `LangGraph` for orchestration and local function calling
- Rule-based checks for abnormalities and nutritional signals

The app is educational and does not provide diagnosis.

## Features

- Upload report files: PDF, TXT, CSV, and common image formats.
- Extract lab values (test name, value, reference range, status).
- Detect potential concerns using a local rules engine.
- Retrieve relevant context from local Chroma knowledge base.
- Require Gemma to call local tools for abnormal-result checks and Chroma retrieval.
- Generate plain-language summary and doctor discussion prompts.
- Keep processing fully local on your machine.

## Project Structure

```text
vaultmd/
├── app.py
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
├── data/
│   ├── chroma/                  # auto-created local vector DB
│   └── knowledge/
│       ├── common_labs.md
│       ├── nutrition_guidance.md
│       └── safety_red_flags.md
└── src/
    ├── __init__.py
    ├── config.py
    ├── models.py
    ├── agents/
    │   ├── __init__.py
    │   ├── graph.py
    │   └── prompts.py
    ├── ingestion/
    │   ├── __init__.py
    │   ├── extractor.py
    │   └── parser.py
    ├── retrieval/
    │   ├── __init__.py
    │   └── vector_store.py
    └── rules/
        ├── __init__.py
        └── flag_engine.py
```

## Prerequisites

1. Python 3.11
2. `uv` installed ([docs](https://docs.astral.sh/uv/getting-started/installation/))
3. [Ollama](https://ollama.com/) installed and running
4. Pulled local models:

```bash
ollama pull gemma4:26b
ollama pull nomic-embed-text
```

5. Optional for image OCR: Install Tesseract binary on macOS:

```bash
brew install tesseract
```

## Setup

From project root:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync --python 3.11
cp .env.example .env
```

## Run

Ensure Ollama is running:

```bash
ollama serve
```

In another terminal:

```bash
uv run streamlit run app.py
```

Open the URL shown by Streamlit (typically `http://localhost:8501`).

## How It Works

1. File parser extracts text from report.
2. Regex extractor detects structured lab values.
3. Report is chunked and indexed in local Chroma report collection.
4. App retrieves relevant report + medical knowledge chunks.
5. LangGraph pipeline runs:
   - `parse_labs`
   - `gemma_tool_call_node`
   - `llm_node`
6. Gemma must call local tools in two sequential steps:
   - `flag_abnormal_results`
   - `retrieve_medical_context`
7. Gemma returns structured JSON, rendered in Streamlit sections.

If Gemma does not return either required tool call, the graph raises an error.
This is intentional because the project is designed to showcase Gemma function
calling, not a deterministic fallback.

## Safety Notes

- The app is informational only and not a substitute for medical care.
- It avoids diagnosis and uses cautious wording.
- Final outputs include doctor follow-up questions and disclaimers.

## Troubleshooting

- `Connection refused` to Ollama:
  - Start Ollama with `ollama serve`.
- Missing model:
  - Run `ollama pull gemma4:26b` and `ollama pull nomic-embed-text`.
- Gemma tool-calling error:
  - Confirm that your Ollama Gemma model supports tool/function calls.
  - Increase `OLLAMA_TIMEOUT_SECONDS` in `.env` if your Mac needs more time.
- No labs extracted:
  - Some report formats are hard to parse; test with clearer PDF/text exports.
- OCR not working:
  - Install `tesseract` system package and retry image upload.
