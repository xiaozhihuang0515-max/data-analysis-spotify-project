import numpy as np


class Preprocessor:

    def normalize(self, df, features):

        df = df.copy()

        for f in features:
            df[f] = (df[f] - df[f].mean()) / df[f].std()

        return df
