# Spotify Audio Explorer

A FastAPI-powered Spotify EDA demo with a browser dashboard and content-based
track recommendations.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. Interactive API documentation is available at
<http://127.0.0.1:8000/docs>.

## Project structure

- `app/`: FastAPI routes and data/embedding logic
- `static/`: browser dashboard
- `visualization/`: reusable visualization data builders
- `eda/embedding_exploration.ipynb`: PCA, clustering, and similarity analysis
- `data/spotify_tracks.csv`: supplied dataset sample

