# Healthcare LLM Pipeline Starter
Starter for a healthcare evidence assistant using retrieval + grounded generation.

## Project layout
- `data/raw/sample_corpus.jsonl`: starter healthcare corpus (toy data)
- `ingest.py`: loads and cleans raw corpus into tabular format
- `index.py`: builds hybrid retrieval index (`data/index/hybrid_index.pkl`; downloads `BAAI/bge-small-en-v1.5` via FastEmbed on first run)
- `query.py`: CLI query for Q&A with citations
- `eval.py`: lightweight retrieval sanity checks
- `app/streamlit_app.py`: local web demo
- `src/healthcare_rag/`: package code

## Quickstart
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional:
```bash
copy .env.example .env
# add OPENAI_API_KEY to .env
```

## Run pipeline
```bash
python ingest.py
python index.py
python query.py --question "What is first-line treatment for stage 1 hypertension with diabetes?"
python eval.py
streamlit run app/streamlit_app.py
```

# To do:
- Replace toy corpus with PubMed / guideline documents
- Add document chunking and metadata filtering by specialty/year
- Tune hybrid weights / RRF constant or add metadata filters (specialty, year)
- Add structured evaluation: recall@k, citation faithfulness, abstention rate
- Log experiments and prompts with MLflow

## Responsibility Note
This project is for educational decision support and not for diagnosis or emergency use. Use source citations and clinician review for all outputs.

