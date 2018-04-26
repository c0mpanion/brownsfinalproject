import time
from collections import deque

import numpy as np


class LinearStrategy:
    def __init__(self, data_frame):
        """Linear strategy approach"""
        print("\nStarting linear strategy...")
        self.df = data_frame

        # Get scores
        start_time = time.time()
        scores = self.score_df()
        print("* Linear strategy completed in {} seconds with {} scores..."
              .format(time.time() - start_time, len(scores)))

        # Test for invalid data or scoring function change
        if scores[1000000] != 0 or scores[1000001] != 0.625:
            raise ValueError("Linear strategy returned an unexpected score, [...{}, {}...].".format(
                scores[1000000], scores[1000001]
            ))

    def print_columns(self):
        print(self.df.columns)

    def print_head(self):
        print(self.df.head())

    def score_df(self):
        killed = self.df['NUMBER OF PERSONS KILLED'].values.astype(np.float32)
        injured = self.df['NUMBER OF PERSONS INJURED'].values.astype(np.float32)

        # Deque is a list-like container with faster performance
        scores = deque()
        for x, y in zip(killed, injured):
            scores.append(self.score_row(x, y))

        # Return list of scores
        return np.array(scores)

    def score_row(self, killed, injured):
        return (((killed * 2.0) + injured) / 8.0) * 5.0
