# Virtual Stylist Prototype – Reflection

## 1) What the prototype does well
- **Clarity & simplicity:** Single `/recommend` endpoint, clear Pydantic schemas, straightforward TF-IDF scoring. Easy to read and reason about.
- **Modularity:** Separated layers (`api/routes`, `models/schemas`, `services/data_loader`, `services/recommender`, `services/filters`, `config`). Filter registry enables drop-in filters without touching core logic.
- **Deterministic, fast startup:** Lightweight TF-IDF over a small catalog/docs; no external services required. Good for demos.
- **Document grounding:** Internal docs are included in scoring via a doc-boost; rationale surfaces doc snippets to explain recommendations.
- **Operational basics:** Health check, config centralization, and venv-based setup make it runnable across Windows/macOS.

## 2) Limitations of the current approach
- **Recommendation accuracy:** TF-IDF is lexical; no semantic understanding of style, silhouettes, or compatibility. No diversification; rankings may cluster similar items.
- **Document use is shallow:** Only a max-similarity doc contributes a scalar boost and snippet. No true retrieval of multiple passages or structured guidance.
- **Logic constraints:** Filters are basic (price/budget/color blocklist). No size/fit/season/inventory considerations; no personalization beyond simple text concat.
- **Scalability:** In-memory pandas + TF-IDF fits small catalogs; not sharded or persisted. Recomputes vectors on startup; no warm cache across workers.
- **Maintainability:** No tests yet. Error handling is minimal. No observability (metrics/log tracing) to diagnose bad recommendations.
- **Data freshness:** Static CSV/docs; no ingestion pipeline, no validation, no deduplication/normalization for tags/colors.

## 3) What to improve with one more week
- **Modeling / retrieval:**
  - Replace TF-IDF with sentence embeddings (e.g., all-MiniLM or E5) for semantic similarity.
  - Add diversification (e.g., MMR) so top picks aren’t duplicates.
  - Retrieve multiple doc passages (top-k) and feed them into rationale instead of single doc boost.
- **Filtering & business logic:**
  - Expand filter registry with size, season, category diversity, brand allow/block lists, inventory/availability, and a style compatibility heuristic.
  - Add a light penalty for disliked colors instead of hard filter to avoid empty results.
- **Data layer:**
  - Introduce a data ingestion script to normalize colors/tags, validate schema, and precompute embeddings to disk.
  - Add config-driven paths and environment overrides; persist corpora/embeddings for faster cold starts.
- **Backend structure & quality:**
  - Add unit tests for filters and recommender; integration tests for `/recommend`.
  - Add logging and minimal metrics (request counts, latency, empty-result rate).
  - Add input validation for price ranges and guardrails on query length.
- **LLM-aware improvements (document-grounded):**
  - Use an embedding retriever over docs + catalog attributes, then feed top passages into a lightweight reranker (e.g., cross-encoder or small LLM) to improve ordering.
  - Optional LLM rationale generation constrained to retrieved facts (preventing hallucination) and bounded to short tokens; keep determinism by caching retrieval results.
  - For large catalogs: precompute embeddings, use an ANN index (FAISS/ScaNN), and a two-stage ranker (vector recall → reranker).

## 4) AI architecture notes (where improvements fit)
- **Retrieval:** Replace TF-IDF with embedding index over products and docs; store in a vector DB or FAISS. Fit: `data_loader` would build/load embeddings; `recommender` would call retriever.
- **Reranking:** Add a second-stage reranker (cross-encoder or small LLM with system prompt) to rescore the top N retrieved products using query + doc snippets. Fit: new `services/reranker.py`, invoked after retrieval before filters.
- **Grounded generation:** Keep generation optional and bounded; use retrieved doc chunks plus product metadata to produce rationales, with strict citations/snippets to avoid drift. Fit: `recommender` calls a `rationale_builder` that consumes retrieved context.
- **Scalability:** Precompute and persist embeddings; warm-load on startup; enable ANN for large catalogs. Use batch retrieval APIs if exposed as a service.

## 5) If more time beyond a week
- Add user profiles/history and collaborative signals (re-rank with personalized priors).
- Add A/B hooks and feature flags for filters/rerankers.
- Add telemetry dashboards (empty-result rate, diversity metrics, doc coverage).

