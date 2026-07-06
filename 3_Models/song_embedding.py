import numpy as np


class SongEmbedding:

    def __init__(self):
        # 未来可以扩展 feature list
        self.features = [
            "danceability",
            "energy",
            "valence",
            "tempo",
            "acousticness",
            "instrumentalness"
        ]

    def to_vector(self, song: dict):
        """
        convert a song into vector
        """

        vector = []

        for f in self.features:
            value = song.get(f, 0)  # 如果没有就给0
            vector.append(value)

        return np.array(vector)

    def batch_to_vectors(self, songs: list):
        """
        mang song--> many vectors
        """

        return np.array([
            self.to_vector(song)
            for song in songs
        ])
