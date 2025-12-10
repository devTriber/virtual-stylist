from __future__ import annotations

from pathlib import Path


# Data paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRODUCTS_PATH = DATA_DIR / "products.csv"
DOCS_DIR = DATA_DIR / "docs"

# Recommendation tuning
PRODUCT_WEIGHT = 0.7
DOC_WEIGHT = 0.3
TOP_K = 4
DOC_SNIPPET_CHARS = 220

