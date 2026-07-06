import pandas as pd


class DataLoader:

    def __init__(self, path="app/data/spotify.csv"):
        self.path = path

    def load(self):
        df = pd.read_csv(self.path)
        return df
