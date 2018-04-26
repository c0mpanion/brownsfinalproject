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
        if scores[1000000][0] != 0 or scores[1000001][0] != 0.625:
            raise ValueError("Linear strategy returned an unexpected score, [...{}, {}...].".format(
                scores[1000000][0], scores[1000001][0]
            ))

    def score_df(self):
        # Create a matrix array
        df = self.df[[
            'SCORE',
            'LATITUDE',
            'LONGITUDE',
            'NUMBER OF PERSONS KILLED',
            'NUMBER OF PERSONS INJURED',
        ]].values.astype(np.float)

        for i in range(len(df)):
            df[i][0] = self.score_row(df[i][3], df[i][4])

        # Return only score and lat/lng
        return df[:, [0, 1, 2]]

    def score_row(self, killed, injured):
        return (((killed * 2.0) + injured) / 8.0) * 5.0
