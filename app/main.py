from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRODUCTS_PATH = DATA_DIR / "products.csv"
DOCS_DIR = DATA_DIR / "docs"


class UserProfile(BaseModel):
    preferred_styles: Optional[List[str]] = None
    disliked_colors: Optional[List[str]] = None
    budget: Optional[float] = None
    notes: Optional[str] = None


class RecommendRequest(BaseModel):
    query: str
    profile: Optional[UserProfile] = None
    max_price: Optional[float] = None


class Recommendation(BaseModel):
    product_id: str
    name: str
    rationale: str


class RecommendResponse(BaseModel):
    items: List[Recommendation]


app = FastAPI(title="Virtual Stylist Prototype")


def load_products() -> pd.DataFrame:
    if not PRODUCTS_PATH.exists():
        raise FileNotFoundError(f"Missing product catalog at {PRODUCTS_PATH}")

    df = pd.read_csv(PRODUCTS_PATH)
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
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Missing docs directory at {DOCS_DIR}")

    docs = []
    for path in DOCS_DIR.glob("*.txt"):
        content = path.read_text(encoding="utf-8").strip()
        if content:
            docs.append({"id": path.stem, "content": content})

    if not docs:
        raise ValueError("No internal documents found.")
    return docs


def build_product_vectorizer(products: pd.DataFrame):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(products["text"])
    return vectorizer, matrix


def build_doc_vectorizer(docs: List[dict]):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([doc["content"] for doc in docs])
    return vectorizer, matrix


def format_profile(profile: Optional[UserProfile]) -> str:
    if profile is None:
        return ""
    parts: List[str] = []
    if profile.preferred_styles:
        parts.append("preferred styles: " + ", ".join(profile.preferred_styles))
    if profile.disliked_colors:
        parts.append("avoid colors: " + ", ".join(profile.disliked_colors))
    if profile.budget:
        parts.append(f"budget under {profile.budget}")
    if profile.notes:
        parts.append(profile.notes)
    return ". ".join(parts)


products_df = load_products()
docs_data = load_docs()
product_vectorizer, product_matrix = build_product_vectorizer(products_df)
doc_vectorizer, doc_matrix = build_doc_vectorizer(docs_data)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    query_text = payload.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query is required.")

    profile_text = format_profile(payload.profile)
    combined_text = (query_text + " " + profile_text).strip()

    try:
        query_vec = product_vectorizer.transform([combined_text])
        product_scores = cosine_similarity(query_vec, product_matrix).ravel()
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Product scoring failed: {exc}")

    try:
        doc_vec = doc_vectorizer.transform([combined_text])
        doc_scores = cosine_similarity(doc_vec, doc_matrix).ravel()
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Document scoring failed: {exc}")

    doc_boost = doc_scores.max() if doc_scores.size else 0.0
    top_doc_idx = int(doc_scores.argmax()) if doc_scores.size else 0
    top_doc = docs_data[top_doc_idx] if docs_data else {"id": "", "content": ""}
    top_doc_snippet = top_doc["content"][:220] + ("..." if len(top_doc["content"]) > 220 else "")

    combined_score = 0.7 * product_scores + 0.3 * doc_boost

    ranked = (
        products_df.assign(score=combined_score)
        .sort_values(by="score", ascending=False)
    )

    if payload.max_price:
        ranked = ranked[ranked["price"] <= payload.max_price]

    if payload.profile and payload.profile.budget:
        ranked = ranked[ranked["price"] <= payload.profile.budget]

    if ranked.empty:
        raise HTTPException(status_code=404, detail="No products match the criteria.")

    top_items = ranked.head(4)
    recommendations: List[Recommendation] = []

    for _, row in top_items.iterrows():
        rationale_parts = [
            f"Matches request for {query_text}",
            f"Color: {row['color']}, Category: {row['category']}",
            f"Doc insight ({top_doc['id']}): {top_doc_snippet}",
        ]
        recommendations.append(
            Recommendation(
                product_id=str(row["product_id"]),
                name=row["name"],
                rationale=" | ".join(rationale_parts),
            )
        )

    return RecommendResponse(items=recommendations)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "products_loaded": len(products_df), "docs_loaded": len(docs_data)}

