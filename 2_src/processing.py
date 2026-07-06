import pandas as pd


class Preprocessor:

    def clean(self, df: pd.DataFrame):

        # drop null
        df = df.dropna()

        # keep only needed features
        features = [
            "danceability",
            "energy",
            "valence",
            "tempo",
            "acousticness",
            "instrumentalness"
        ]

        return df[features]
