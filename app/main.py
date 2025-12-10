from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.services import data_loader
from app.services.filters import default_filters
from app.services.recommender import Recommender


app = FastAPI(title="Virtual Stylist Prototype")


@app.on_event("startup")
def startup_event() -> None:
    products_corpus, docs_corpus = data_loader.load_corpora()
    app.state.recommender = Recommender(
        products=products_corpus,
        docs=docs_corpus,
        filters=default_filters,
    )
    app.state.products_loaded = len(products_corpus.df)
    app.state.docs_loaded = len(docs_corpus.docs)


app.include_router(router)

