from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from fastapi import HTTPException
from sklearn.metrics.pairwise import cosine_similarity

from app import config
from app.models.schemas import RecommendRequest, RecommendResponse, Recommendation
from app.services.data_loader import DocsCorpus, ProductCorpus
from app.services.filters import FilterRegistry


@dataclass
class Rationale:
    filter_notes: List[str]
    doc_id: str
    doc_snippet: str
    query_text: str

    def to_text(self, product_row: pd.Series) -> str:
        parts = [
            f"Matches request for {self.query_text}",
            f"Color: {product_row['color']}, Category: {product_row['category']}",
            f"Doc insight ({self.doc_id}): {self.doc_snippet}",
        ]
        if self.filter_notes:
            parts.append("Filters: " + ", ".join(self.filter_notes))
        return " | ".join(parts)


class Recommender:
    def __init__(self, products: ProductCorpus, docs: DocsCorpus, filters: FilterRegistry):
        self.products = products
        self.docs = docs
        self.filters = filters

    def recommend(self, payload: RecommendRequest) -> RecommendResponse:
        query_text = payload.query.strip()
        if not query_text:
            raise HTTPException(status_code=400, detail="Query is required.")

        combined_text = self._combine_query_and_profile(payload, query_text)
        product_scores = self._score_products(combined_text)
        doc_scores = self._score_docs(combined_text)

        doc_boost, top_doc = self._doc_boost(doc_scores)
        doc_snippet = self._snippet(top_doc["content"])

        combined_score = config.PRODUCT_WEIGHT * product_scores + config.DOC_WEIGHT * doc_boost

        ranked = (
            self.products.df.assign(score=combined_score)
            .sort_values(by="score", ascending=False)
        )

        ranked, filter_notes = self.filters.apply(ranked, payload)
        if ranked.empty:
            raise HTTPException(status_code=404, detail="No products match the criteria.")

        top_items = ranked.head(config.TOP_K)
        rationale = Rationale(
            filter_notes=filter_notes,
            doc_id=top_doc["id"],
            doc_snippet=doc_snippet,
            query_text=query_text,
        )

        recommendations: List[Recommendation] = [
            Recommendation(
                product_id=str(row["product_id"]),
                name=row["name"],
                rationale=rationale.to_text(row),
            )
            for _, row in top_items.iterrows()
        ]

        return RecommendResponse(items=recommendations)

    def _combine_query_and_profile(self, payload: RecommendRequest, query_text: str) -> str:
        profile = payload.profile
        parts: List[str] = [query_text]
        if profile:
            if profile.preferred_styles:
                parts.append("preferred styles: " + ", ".join(profile.preferred_styles))
            if profile.disliked_colors:
                parts.append("avoid colors: " + ", ".join(profile.disliked_colors))
            if profile.budget:
                parts.append(f"budget under {profile.budget}")
            if profile.notes:
                parts.append(profile.notes)
        return " ".join(parts).strip()

    def _score_products(self, text: str):
        try:
            return cosine_similarity(
                self.products.vectorizer.transform([text]), self.products.matrix
            ).ravel()
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"Product scoring failed: {exc}")

    def _score_docs(self, text: str):
        try:
            return cosine_similarity(
                self.docs.vectorizer.transform([text]), self.docs.matrix
            ).ravel()
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"Document scoring failed: {exc}")

    def _doc_boost(self, doc_scores) -> Tuple[float, dict]:
        if doc_scores.size == 0:
            return 0.0, {"id": "", "content": ""}
        idx = int(doc_scores.argmax())
        return float(doc_scores.max()), self.docs.docs[idx]

    def _snippet(self, content: str) -> str:
        if len(content) <= config.DOC_SNIPPET_CHARS:
            return content
        return content[: config.DOC_SNIPPET_CHARS] + "..."

