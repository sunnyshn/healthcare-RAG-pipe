"""Fetch PubMed abstracts and append them to the raw corpus JSONL.

Usage
-----
    python scripts/fetch_pubmed.py --query "type 2 diabetes management" --max 50 --specialty endocrinology
    python scripts/fetch_pubmed.py --query "sepsis treatment guidelines" --max 30 --specialty "infectious disease"
    python scripts/fetch_pubmed.py --query "breast cancer immunotherapy" --max 100 --specialty oncology

After fetching, rebuild the search index:
    python scripts/ingest.py && python scripts/index.py

Rate limits (NCBI E-utilities)
-------------------------------
    Without NCBI_API_KEY : 3 requests/sec
    With    NCBI_API_KEY : 10 requests/sec
    Set NCBI_API_KEY in your .env file to increase throughput.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, List, Optional

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from healthcare_rag.config import RAW_PATH

ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
FETCH_BATCH = 100  # max PMIDs per efetch call
_RATE_DEFAULT = 0.4  # ~3 req/sec (no key)
_RATE_WITH_KEY = 0.12  # ~10 req/sec (with key)


# ---------------------------------------------------------------------------
# NCBI E-utilities helpers
# ---------------------------------------------------------------------------


def _api_key() -> Optional[str]:
    return os.getenv("NCBI_API_KEY") or None


def _rate_delay() -> float:
    return _RATE_WITH_KEY if _api_key() else _RATE_DEFAULT


def _base_params() -> dict:
    params: dict = {}
    key = _api_key()
    if key:
        params["api_key"] = key
    return params


def esearch(query: str, max_results: int) -> List[str]:
    """Return up to *max_results* PubMed IDs matching *query*."""
    params = {
        **_base_params(),
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    r = requests.get(f"{ENTREZ_BASE}/esearch.fcgi", params=params, timeout=15)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


def _efetch_xml(pmids: List[str]) -> str:
    params = {
        **_base_params(),
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "xml",
        "retmode": "xml",
    }
    r = requests.get(f"{ENTREZ_BASE}/efetch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    return r.text


def _parse_articles(xml_text: str) -> Iterator[dict]:
    """Yield one dict per PubmedArticle that has a non-empty abstract."""
    root = ET.fromstring(xml_text)
    for article in root.iter("PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None or not pmid_el.text:
            continue
        pmid = pmid_el.text.strip()

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        # Structured abstracts have multiple <AbstractText> sections.
        abstract_parts = [
            "".join(el.itertext()).strip()
            for el in article.findall(".//AbstractText")
        ]
        abstract = " ".join(filter(None, abstract_parts))
        if not abstract:
            continue  # skip articles without an abstract

        # Publication year: prefer PubDate/Year, fall back to MedlineDate prefix.
        year: int = 0
        for tag in ("Year", "MedlineDate"):
            el = article.find(f".//PubDate/{tag}")
            if el is not None and el.text:
                try:
                    year = int(el.text[:4])
                    break
                except ValueError:
                    pass

        journal_el = article.find(".//Journal/Title")
        source = journal_el.text.strip() if journal_el is not None else "PubMed"

        yield {
            "doc_id": f"pmid-{pmid}",
            "title": title,
            "text": abstract,
            "year": year,
            "source": source,
        }


def fetch_all(query: str, max_results: int) -> List[dict]:
    """Search PubMed and return parsed article dicts."""
    pmids = esearch(query, max_results)
    if not pmids:
        return []

    delay = _rate_delay()
    docs: List[dict] = []
    batches = [pmids[i : i + FETCH_BATCH] for i in range(0, len(pmids), FETCH_BATCH)]

    for batch in tqdm(batches, desc="Fetching abstracts", unit="batch"):
        xml_text = _efetch_xml(batch)
        docs.extend(_parse_articles(xml_text))
        time.sleep(delay)

    return docs


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------


def _load_existing_ids(path: Path) -> set:
    ids: set = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["doc_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch PubMed abstracts and append to the corpus JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query", required=True, help="PubMed search query string")
    parser.add_argument(
        "--max",
        type=int,
        default=50,
        metavar="N",
        help="Maximum number of articles to fetch",
    )
    parser.add_argument(
        "--specialty",
        default="general",
        metavar="SPECIALTY",
        help="Specialty label assigned to every fetched document",
    )
    args = parser.parse_args()

    key = _api_key()
    if key:
        print("NCBI API key detected — using 10 req/s rate limit.")
    else:
        print(
            "No NCBI_API_KEY found — using public rate limit (3 req/s). "
            "Add NCBI_API_KEY to .env for faster fetching."
        )

    print(f'\nSearching PubMed: "{args.query}" (max {args.max} results)...')
    docs = fetch_all(args.query, args.max)
    print(f"Found {len(docs)} articles with abstracts.")

    existing_ids = _load_existing_ids(RAW_PATH)
    new_docs = [d for d in docs if d["doc_id"] not in existing_ids]
    skipped = len(docs) - len(new_docs)
    if skipped:
        print(f"Skipped {skipped} duplicate(s) already in corpus.")

    if not new_docs:
        print("No new documents to add.")
        return

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_PATH.open("a", encoding="utf-8") as f:
        for doc in new_docs:
            doc["specialty"] = args.specialty
            f.write(json.dumps(doc) + "\n")

    print(f"\nAppended {len(new_docs)} new document(s) to {RAW_PATH}")
    print("Rebuild the index with:\n  python scripts/ingest.py && python scripts/index.py")


if __name__ == "__main__":
    main()
