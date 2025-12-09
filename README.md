# Virtual Stylist – Prototype Backend

A minimal FastAPI backend that recommends products from a small catalog using both product metadata and internal styling documents. The goal is to demonstrate end-to-end reasoning, not production readiness.

## Setup (cross-platform)

Prereqs: Python 3.11+ is recommended so wheels are available for `pandas` and `scikit-learn` on Windows/macOS (avoids compiling).

Create a virtual environment:
- Windows (PowerShell):
```
py -3.11 -m venv .venv
.\.venv\Scripts\activate
```
- macOS/Linux:
```
python3 -m venv .venv
source .venv/bin/activate
```

Install deps:
```
pip install --upgrade pip
pip install -r requirements.txt
```

Run the API:
```
python -m uvicorn app.main:app --reload
```

The service starts at `http://127.0.0.1:8000`.

## Test the endpoint

POST `http://127.0.0.1:8000/recommend`

Example request:
```
curl -X POST http://127.0.0.1:8000/recommend ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"smart casual outfit for office in fall\", \"profile\": {\"preferred_styles\": [\"minimal\"], \"budget\": 120}}"
```

Expected response shape (values will vary):
```
{
  "items": [
    {
      "product_id": "P1004",
      "name": "Wide-Leg Trousers",
      "rationale": "Matches request for smart casual outfit for office in fall | Color: black, Category: bottoms | Doc insight (styling_guide): When recommending outfits..."
    },
    ...
  ]
}
```

## Data sources

- Product catalog: `data/products.csv`
- Internal documents: `data/docs/*.txt`

## Notes

- Recommendations are vector-based (TF-IDF) over product text and internal docs, combined with the user query and optional profile text.
- Use `/health` to confirm data loads: `http://127.0.0.1:8000/health`.

## Push to your git repo

From the project root:
```
git init
git add .
git commit -m "Add virtual stylist prototype"
git remote add origin <your_repo_url>
git push -u origin main
```

