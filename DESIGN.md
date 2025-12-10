# Virtual Stylist Prototype – Design Notes

## Overview
- Goal: recommend 3–4 products for a user query by combining product catalog metadata (CSV) with internal styling documents.
- Stack: FastAPI, pandas, scikit-learn TF-IDF, Python 3.11+.
- Entry point: `/recommend` (POST) plus `/health`.

## Architecture
- `app/main.py`: FastAPI app setup, startup loading, router include.
- `app/api/routes.py`: HTTP routes; pulls dependencies from app state.
- `app/models/schemas.py`: Pydantic request/response models.
- `app/services/data_loader.py`: Loads catalog/docs, builds TF-IDF vectorizers and matrices.
- `app/services/recommender.py`: Scores products, merges doc signals, builds rationales, applies filters.
- `app/services/filters.py`: Composable filter pipeline (price, profile budget, disliked colors), easy to extend.
- `app/config.py`: Paths, weights, top_k, and other tunables.
- Data: `data/products.csv`, `data/docs/*.txt` (plain text).

### Request/Response
- Request: `query` (required), optional `profile` (preferred_styles, disliked_colors, budget, notes), optional `max_price`.
- Response: `items` array with `product_id`, `name`, `rationale`.

### Scoring Flow
1) Format query + profile text.
2) TF-IDF similarity against product corpus → `product_scores`.
3) TF-IDF similarity against doc corpus → `doc_scores`; derive `doc_boost = max(doc_scores)`.
4) Combined score = `0.7 * product_scores + 0.3 * doc_boost` (configurable).
5) Rank, apply filter pipeline, take top_k (default 4).
6) Build rationale with query match, product attributes, and top doc snippet.

### Filter Pipeline (Extensible)
- Filters are callables `(df, payload, context) -> (df, notes)`; they are chained in `default_filters`.
- Current filters: `price_filter`, `profile_budget_filter`, `disliked_colors_filter`.
- To add a filter: implement a function in `filters.py` and append to `default_filters`. Keep them pure (no globals) and return the same schema.

## Running
- Create/activate venv; `pip install -r requirements.txt`; run `python -m uvicorn app.main:app --reload`.
- Health: `GET /health`.
- Example recommend request is in `README.md`.

## Extensibility Ideas
- Swap TF-IDF for sentence embeddings.
- Add diversification (e.g., avoid same category repetition).
- Add size/season/brand/inventory filters via pipeline.
- Externalize weights/top_k to env vars.
- Persist models or cache in memory across workers.

## Error Handling
- 400 on missing query.
- 404 when filters empty the set.
- 500 guarded around vectorization failures.

## Testing (suggested quick checks)
- Unit: filter functions with small DataFrames.
- Integration: call `/recommend` with/without budgets; ensure 404 on over-constrained filters.
- Smoke: `/health` returns counts > 0.

