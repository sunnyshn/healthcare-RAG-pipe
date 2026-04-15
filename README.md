# Healthcare LLM Pipeline Starter
Starter for a healthcare evidence assistant using retrieval + grounded generation.

## Project layout
- `data/raw/sample_corpus.jsonl`: healthcare corpus (grows as you fetch from PubMed)
- `scripts/fetch_pubmed.py`: fetch PubMed abstracts and append to corpus
- `scripts/ingest.py`: loads and cleans raw corpus into tabular format
- `scripts/index.py`: builds hybrid retrieval index (`data/index/hybrid_index.pkl`; downloads `BAAI/bge-small-en-v1.5` via FastEmbed on first run)
- `scripts/query.py`: CLI query for Q&A with citations
- `scripts/eval.py`: retrieval eval with recall@1 and recall@3
- `app/streamlit_app.py`: local web demo
- `src/healthcare_rag/`: package code
- `tests/`: unit tests (pytest)

## Quickstart
```bash
python -m venv .venv
# Linux / WSL:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set up your API keys:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (required for LLM answers)
# Optionally add NCBI_API_KEY for faster PubMed fetching
```

## Expand the corpus with PubMed
```bash
# Fetch abstracts by topic and specialty
python scripts/fetch_pubmed.py --query "type 2 diabetes management" --max 50 --specialty endocrinology
python scripts/fetch_pubmed.py --query "atrial fibrillation anticoagulation" --max 50 --specialty cardiology
python scripts/fetch_pubmed.py --query "major depressive disorder treatment" --max 50 --specialty psychiatry
python scripts/fetch_pubmed.py --query "community acquired pneumonia antibiotics" --max 50 --specialty "infectious disease"

# Rebuild the index after fetching
python scripts/ingest.py
python scripts/index.py
```

## Run pipeline
```bash
python scripts/ingest.py
python scripts/index.py
python scripts/query.py --question "What is first-line treatment for stage 1 hypertension with diabetes?"

# Filter by specialty or year range
python scripts/query.py --question "diabetes drug options" --specialty endocrinology --year_min 2020

python scripts/eval.py
streamlit run app/streamlit_app.py
```

## Ingest with chunking (for long documents)
```bash
python scripts/ingest.py --chunk --chunk-size 450 --overlap 75
python scripts/index.py
```

## Docker
Run the entire app in a container with no local setup required:
```bash
cp .env.example .env   # add your OPENAI_API_KEY
docker compose up --build
```
Open `http://localhost:8501` in your browser. The corpus and index are persisted in `data/` via a volume mount, so they survive container restarts.

To run only the tests inside Docker:
```bash
docker compose run --rm app python -m pytest tests/ -v
```

## Tests
```bash
python -m pytest tests/ -v
```

## To do
- Log experiments and prompts with MLflow
- Add citation faithfulness and abstention rate to eval
- Add metadata filtering by MeSH terms

## Responsibility Note
This project is for educational decision support and not for diagnosis or emergency use. Use source citations and clinician review for all outputs.

