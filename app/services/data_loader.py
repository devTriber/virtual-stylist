from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from app import config


@dataclass
class DocsCorpus:
    docs: List[dict]
    vectorizer: TfidfVectorizer
    matrix: any  # sparse matrix


@dataclass
class ProductCorpus:
    df: pd.DataFrame
    vectorizer: TfidfVectorizer
    matrix: any  # sparse matrix


def load_products() -> pd.DataFrame:
    if not config.PRODUCTS_PATH.exists():
        raise FileNotFoundError(f"Missing product catalog at {config.PRODUCTS_PATH}")

    df = pd.read_csv(config.PRODUCTS_PATH)
    if df.empty:
        raise ValueError("Product catalog is empty.")

    df["text"] = (
        df["name"].fillna("")
        + " "
        + df["category"].fillna("")
        + " "
        + df["color"].fillna("")
        + " "
        + df["description"].fillna("")
        + " "
        + df.get("tags", "").fillna("")
    )
    return df


def load_docs() -> List[dict]:
    if not config.DOCS_DIR.exists():
        raise FileNotFoundError(f"Missing docs directory at {config.DOCS_DIR}")

    docs = []
    for path in config.DOCS_DIR.glob("*.txt"):
        content = path.read_text(encoding="utf-8").strip()
        if content:
            docs.append({"id": path.stem, "content": content})

    if not docs:
        raise ValueError("No internal documents found.")
    return docs


def build_product_corpus(df: pd.DataFrame) -> ProductCorpus:
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(df["text"])
    return ProductCorpus(df=df, vectorizer=vectorizer, matrix=matrix)


def build_docs_corpus(docs: List[dict]) -> DocsCorpus:
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([doc["content"] for doc in docs])
    return DocsCorpus(docs=docs, vectorizer=vectorizer, matrix=matrix)


def load_corpora() -> Tuple[ProductCorpus, DocsCorpus]:
    products = load_products()
    docs = load_docs()
    return build_product_corpus(products), build_docs_corpus(docs)

