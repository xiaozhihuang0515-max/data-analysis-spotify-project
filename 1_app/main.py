from __future__ import annotations

from collections import Counter
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.data import FEATURES, cosine_distance, embedding, tracks
from visualization.charts import genre_overview, popularity_histogram, scatter_sample

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(
    title="Spotify Audio Explorer",
    description="Interactive EDA and content-based track similarity demo.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "tracks": len(tracks())}


@app.get("/api/summary")
def summary() -> dict:
    rows = tracks()
    return {
        "tracks": len(rows),
        "artists": len({row["artists"] for row in rows}),
        "genres": len({row["track_genre"] for row in rows}),
        "avg_popularity": round(
            sum(float(row["popularity"]) for row in rows) / len(rows), 1
        ),
        "top_genres": Counter(row["track_genre"] for row in rows).most_common(8),
    }


@app.get("/api/visualizations")
def visualizations() -> dict:
    rows = tracks()
    return {
        "genres": genre_overview(rows),
        "popularity": popularity_histogram(rows),
        "scatter": scatter_sample(rows),
    }


@app.get("/api/tracks")
def search_tracks(
    q: str = Query(default="", max_length=100),
    genre: str | None = None,
    limit: int = Query(default=12, ge=1, le=50),
) -> list[dict]:
    needle = q.casefold().strip()
    results = [
        row
        for row in tracks()
        if (not needle or needle in f"{row['track_name']} {row['artists']}".casefold())
        and (not genre or row["track_genre"] == genre)
    ]
    results.sort(key=lambda row: float(row["popularity"]), reverse=True)
    return results[:limit]


@app.get("/api/recommend/{track_id}")
def recommend(track_id: str, limit: int = Query(default=8, ge=1, le=20)) -> dict:
    rows = tracks()
    source = next((row for row in rows if row["track_id"] == track_id), None)
    if source is None:
        raise HTTPException(status_code=404, detail="Track not found")
    source_vector = embedding(source)
    matches = []
    for candidate in rows:
        if candidate["track_id"] == track_id:
            continue
        distance = cosine_distance(source_vector, embedding(candidate))
        matches.append(
            {
                "track_id": candidate["track_id"],
                "track_name": candidate["track_name"],
                "artists": candidate["artists"],
                "genre": candidate["track_genre"],
                "similarity": round(1 - distance, 4),
            }
        )
    matches.sort(key=lambda item: item["similarity"], reverse=True)
    return {
        "source": {
            "track_id": source["track_id"],
            "track_name": source["track_name"],
            "artists": source["artists"],
            "features": {feature: source[feature] for feature in FEATURES},
        },
        "recommendations": matches[:limit],
    }

