from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import pandas as pd

from app.models.schemas import RecommendRequest


FilterFunc = Callable[[pd.DataFrame, RecommendRequest], Tuple[pd.DataFrame, List[str]]]


@dataclass
class FilterRegistry:
    filters: List[FilterFunc]

    def apply(self, df: pd.DataFrame, payload: RecommendRequest) -> Tuple[pd.DataFrame, List[str]]:
        notes: List[str] = []
        current = df
        for flt in self.filters:
            current, flt_notes = flt(current, payload)
            notes.extend(flt_notes)
            if current.empty:
                break
        return current, notes


def price_filter(df: pd.DataFrame, payload: RecommendRequest) -> Tuple[pd.DataFrame, List[str]]:
    if payload.max_price is None:
        return df, []
    filtered = df[df["price"] <= payload.max_price]
    return filtered, ["within max_price"] if not filtered.empty else []


def profile_budget_filter(df: pd.DataFrame, payload: RecommendRequest) -> Tuple[pd.DataFrame, List[str]]:
    budget = payload.profile.budget if payload.profile else None
    if budget is None:
        return df, []
    filtered = df[df["price"] <= budget]
    return filtered, ["within profile budget"] if not filtered.empty else []


def disliked_colors_filter(df: pd.DataFrame, payload: RecommendRequest) -> Tuple[pd.DataFrame, List[str]]:
    disliked = (payload.profile.disliked_colors or []) if payload.profile else []
    if not disliked:
        return df, []
    filtered = df[~df["color"].str.lower().isin([c.lower() for c in disliked])]
    return filtered, ["avoids disliked colors"] if not filtered.empty else []


default_filters = FilterRegistry(
    filters=[
        price_filter,
        profile_budget_filter,
        disliked_colors_filter,
    ]
)

