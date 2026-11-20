"""Open the Space.

The library starts loading before the page is served, so a visitor sees the
app while the models are still opening.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from healthcare_rag.boot import start

def main() -> None:
    start()
    from streamlit.web import cli

    sys.argv = [
        "streamlit",
        "run",
        str(ROOT / "app" / "streamlit_app.py"),
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.headless=true",
    ]
    cli.main()


if __name__ == "__main__":
    main()
