# Healthcare LLM Pipeline Starter
Starter for a healthcare evidence assistant using retrieval + grounded generation.

## Project layout
- `data/raw/sample_corpus.jsonl`: healthcare corpus (grows as you fetch from PubMed)
- `scripts/fetch_pubmed.py`: fetch PubMed abstracts and append to corpus
- `scripts/ingest.py`: loads and cleans raw corpus into tabular format
- `scripts/index.py`: builds hybrid retrieval index (`data/index/hybrid_index.pkl`; downloads `BAAI/bge-small-en-v1.5` via FastEmbed on first run)
- `scripts/query.py`: CLI query for Q&A with citations (add `--rerank` for cross-encoder re-ranking)
- `scripts/eval.py`: retrieval eval with recall@1 and recall@3
- `app/streamlit_app.py`: local web demo
- `src/healthcare_rag/`: package code (retriever, reranker, generator, faithfulness, data I/O, config)

## How retrieval works
1. **Stage 1 — hybrid retrieval.** A TF-IDF (lexical) and a dense embedding
   (semantic) retriever each rank the corpus; their rankings are merged with
   Reciprocal Rank Fusion (RRF).
2. **Stage 2 — cross-encoder re-ranking (optional).** With `--rerank`, the
   hybrid stage returns a wider candidate pool (`RERANK_CANDIDATES`, default 20)
   and a cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`) scores each
   (query, passage) pair *jointly* to reorder the final top-k. This trades a
   little latency for more precise top results.
3. **Grounded generation.** The LLM answers strictly from the retrieved
   passages and abstains when evidence is insufficient.
4. **Citation faithfulness (eval + UI).** Each `[n]` citation in the answer is
   checked against the passage it points to using the dense encoder: we report
   how many claims are cited (coverage), how many citations are actually
   on-topic (support), and any citations pointing to a passage that wasn't
   retrieved (invalid). Embedding similarity is a proxy for support, not a
   formal entailment check.

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

# Add cross-encoder re-ranking for more precise top results
python scripts/query.py --question "diabetes drug options" --rerank

# Filter by specialty or year range
python scripts/query.py --question "diabetes drug options" --specialty endocrinology --year_min 2020

python scripts/eval.py            # retrieval recall + abstention + citation faithfulness
python scripts/eval.py --rerank   # same metrics with cross-encoder re-ranking
streamlit run app/streamlit_app.py
```

## Ingest with chunking (for long documents)
```bash
python scripts/ingest.py --chunk --chunk-size 450 --overlap 75
python scripts/index.py
```

## Experiment tracking with MLflow
Every `eval.py` run logs its config (params) and scores (metrics) to MLflow so
you can compare runs, e.g. re-ranking on vs off:
```bash
python scripts/eval.py                       # logs run "baseline"
python scripts/eval.py --rerank              # logs run "rerank"
python scripts/eval.py --no-mlflow           # skip logging
python scripts/eval.py --run-name my-experiment

# Browse runs in the MLflow UI
mlflow ui --backend-store-uri ./mlruns
# then open http://localhost:5000
```
Logged **params**: rerank on/off, top-k, embedding/rerank model, thresholds,
generation mode, corpus size. Logged **metrics**: recall@1, recall@3,
abstention rate, citation coverage/support, invalid-citation rate, mean
support. The eval CSV is attached as a run artifact. Override the tracking
location with `MLFLOW_TRACKING_URI` / `MLFLOW_EXPERIMENT_NAME`.

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
- Add metadata filtering by MeSH terms
- Expand corpus diversity and the answerable/unanswerable eval sets

## Responsibility Note
This project is for educational decision support and not for diagnosis or emergency use. Use source citations and clinician review for all outputs.

