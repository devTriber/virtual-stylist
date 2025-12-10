from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request

from app.models.schemas import RecommendRequest, RecommendResponse
from app.services.recommender import Recommender

router = APIRouter()


def get_recommender(request: Request) -> Recommender:
    recommender: Recommender = getattr(request.app.state, "recommender", None)
    if recommender is None:
        raise HTTPException(status_code=500, detail="Recommender not initialized.")
    return recommender


@router.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest, recommender: Recommender = Depends(get_recommender)) -> RecommendResponse:
    return recommender.recommend(payload)


@router.get("/health")
def health(request: Request) -> dict:
    products_loaded = getattr(request.app.state, "products_loaded", 0)
    docs_loaded = getattr(request.app.state, "docs_loaded", 0)
    return {"status": "ok", "products_loaded": products_loaded, "docs_loaded": docs_loaded}

