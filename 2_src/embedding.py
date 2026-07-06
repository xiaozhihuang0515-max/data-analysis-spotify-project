import numpy as np


class EmbeddingModel:

    def __init__(self, features):
        self.features = features

    def vectorize(self, row):
        return np.array([row[f] for f in self.features])
