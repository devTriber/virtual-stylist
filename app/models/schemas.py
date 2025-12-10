from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


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

