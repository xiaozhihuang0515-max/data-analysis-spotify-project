import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.src.embedding import EmbeddingModel


class Recommender:

    def __init__(self, features):
        self.embedder = EmbeddingModel(features)

    def build_user_profile(self, liked_songs):
        vectors = [self.embedder.vectorize(song) for song in liked_songs]
        return np.mean(vectors, axis=0)

    def recommend(self, user_vector, catalog, top_k=10):

        song_vectors = [
            self.embedder.vectorize(song) for song in catalog
        ]

        similarities = cosine_similarity(
            [user_vector],
            song_vectors
        )[0]

        results = []

        for i, score in enumerate(similarities):
            results.append({
                "track_id": catalog[i]["track_id"],
                "score": float(score)
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]
