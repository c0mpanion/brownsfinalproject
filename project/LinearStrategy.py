import numpy as np


class LinearStrategy:
    """Linear strategy approach"""
    def __init__(self, data_frame):
        print("Starting linear strategy...")
        self.df = data_frame
        self.print_columns()
        self.print_head()

    def print_columns(self):
        print(self.df.columns)

    def print_head(self):
        print(self.df.head())
