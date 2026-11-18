---
sdk: docker
app_port: 8501
---

# Healthcare Evidence Assistant

A small demo that answers clinical questions from a library of PubMed abstracts. The reply points at the passages it used, and those links open the paper when we have one. If the library does not have a close match, the app says so instead of guessing.

It is for learning and for showing how answers can stay tied to sources. It is not for diagnosis or emergency care.

The live demo is at [sunnyshn-healthcare-diagnostic-assistant.hf.space](https://sunnyshn-healthcare-diagnostic-assistant.hf.space).

The few lines at the top of this file tell Hugging Face Spaces how to start the app. You can ignore them when you run the project on your own computer.

## What you can do in the app

- **Ask.** Type a question, or try one of the examples. The answer cites the passages it used.
- **Model monitoring.** After a question, this tab shows how the visit is going: how long answers take, how often the app declines, and whether the citations match the text. The question itself is not saved. The charts belong to this visit, and a refresh starts a new sample.
- **Corpus.** See what is in the library, fetch more abstracts, and rebuild the search index.

## How an answer is made

1. **Search.** The app looks for passages by the words in your question and by meaning, then blends the two lists.
2. **A closer read.** A second pass reads the question together with each promising passage and keeps the closest ones. This is on by default. Turn it off in the sidebar for a faster, rougher search.
3. **A careful reply.** The answer is written only from what was found. When nothing is close enough, the app declines.
4. **A citation check.** Each numbered citation is compared with the passage it points to, so you can see whether that source really fits the sentence.

## Run it on your computer

```bash
python scripts/ingest.py
python scripts/index.py
streamlit run app/streamlit_app.py
```

The first index build downloads a small search model and can take a few minutes. Then open http://localhost:8501.

You can ask from the terminal too:

```bash
python scripts/query.py --question "What is first-line treatment for stage 1 hypertension with diabetes?"
python scripts/query.py --question "diabetes drug options" --specialty endocrinology --year_min 2020
```

## Add more abstracts

Pick a topic and a specialty label, then rebuild the index so the new papers are searchable.

```bash
python scripts/fetch_pubmed.py --query "type 2 diabetes management" --max 50 --specialty endocrinology
python scripts/ingest.py
python scripts/index.py
```

Labels already in the library include cardiology, psychiatry, endocrinology, infectious disease, and general.

For a long paper, split it into overlapping pieces before indexing:

```bash
python scripts/ingest.py --chunk --chunk-size 450 --overlap 75
python scripts/index.py
```

## Looking at quality over time

`scripts/eval.py` asks a fixed set of questions and saves the scores, so you can see whether a change helped.

```bash
python scripts/eval.py
python scripts/eval.py --rerank
mlflow ui --backend-store-uri ./mlruns
```

Open http://localhost:5000 to browse those runs. Add `--no-mlflow` to skip the log, or `--run-name my-experiment` to name it.

Model monitoring in the app is a separate log of live questions. To pause it:

```bash
MLFLOW_MONITOR=0 streamlit run app/streamlit_app.py
```

## Run it with Docker

```bash
docker compose up --build
```

Open http://localhost:8501. The library in `data/` stays on your machine when the container stops, and so do the logs in `mlruns/`.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

Inside Docker:

```bash
docker compose run --rm app sh -c "pip install -r requirements-dev.txt && python -m pytest tests/ -v"
```

## Where things live

- `app/streamlit_app.py` is the demo.
- `data/raw/sample_corpus.jsonl` is the library of abstracts.
- `scripts/` fetches papers, builds the index, and asks from the terminal.
- `src/healthcare_rag/` is the code behind the app.

## A note on use

Read the cited sources, and have a clinician review anything that would affect care. This project is educational decision support, not a diagnostic tool, and it is not for emergencies.
