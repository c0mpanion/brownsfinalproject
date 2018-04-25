import multiprocessing
import time
from collections import deque

import numpy as np


class ParallelStrategy:
    LABEL_KILLED = "NUMBER OF PERSONS KILLED"
    LABEL_INJURED = "NUMBER OF PERSONS INJURED"
    ZIP_CODE = "ZIP_CODE"

    def __init__(self, data_frame):
        """Parallel strategy approach"""
        print("Starting parallel strategy...")
        self.df = data_frame
        self.df_groups = self.df.groupby('ZIP CODE')

        # Get scores
        start_time = time.time()
        scores = self.score_df()
        print("Parallel strategy completed in {} seconds...".format(time.time() - start_time))

    def print_columns(self):
        print(self.df.columns)

    def score_df(self):
        """Convert data frame into groups and run each group in a thread"""
        killed = self.df_groups[self.LABEL_KILLED].apply(list)
        injured = self.df_groups[self.LABEL_INJURED].apply(list)

        # Multiple threads will use this, how to handle race conditions?
        shared_dest = deque()

        # Shared variables for each thread to communicate
        manager = multiprocessing.Manager()
        shared_dict = manager.dict()

        jobs = []
        for i in range(len(killed)):
            p = multiprocessing.Process(target=self.score_group, args=(i, shared_dict, killed[i], injured[i], ))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        print shared_dict.values()

        # Convert deque to array
        dest = np.array(shared_dest)

    def score_group(self, process_num, shared_dict, killed, injured):
        """Score groups of collisions by zip code"""
        # Deque is a list-like container with faster performance
        scores = deque()

        # Iterate between two arrays
        for x, y in zip(killed, injured):
            scores.append(self.score_row(x, y))

        # Append to destination deque
        # shared_dest.append(scores)
        shared_dict[process_num] = np.array(scores)
        return shared_dict

    def score_row(self, killed, injured):
        """Score individual collision row"""
        return (((killed * 2) + injured) / 5) * 5
