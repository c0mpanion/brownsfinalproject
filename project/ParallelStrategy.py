import multiprocessing
import time
from collections import deque
from itertools import chain

import numpy as np


class ParallelStrategy:
    LABEL_KILLED = "NUMBER OF PERSONS KILLED"
    LABEL_INJURED = "NUMBER OF PERSONS INJURED"
    ZIP_CODE = "ZIP_CODE"

    def __init__(self, data_frame):
        """Parallel strategy approach"""
        print("\nStarting parallel strategy...")

        # Get data frame and fill empty zip code columns
        self.df = data_frame
        self.df['ZIP CODE'] = self.df['ZIP CODE'].fillna(-1)
        self.df_groups = self.df.groupby('ZIP CODE')

        # Get scores
        start_time = time.time()
        scores = self.score_df()
        print("* Parallel strategy completed in {} seconds with {} scores..."
              .format(time.time() - start_time, len(scores)))

    def print_columns(self):
        print(self.df.columns)

    def score_df(self):
        """Convert data frame into groups and run each group in a thread"""
        killed = self.df_groups[self.LABEL_KILLED].apply(list)
        injured = self.df_groups[self.LABEL_INJURED].apply(list)

        # Shared variables for each thread to communicate
        manager = multiprocessing.Manager()
        shared_dict = manager.dict()

        # List of jobs
        jobs = []

        # Iterate through all zip codes and create a process
        for i in range(len(killed)):
            p = multiprocessing.Process(target=self.score_group, args=(i, shared_dict, killed[i], injured[i]))
            jobs.append(p)
            p.start()

        # Wait for all jobs to finish
        for proc in jobs:
            proc.join()

        # Order is not important, merge all scores from dictionary
        values = chain.from_iterable(shared_dict.values())
        return np.array(list(values))

    def score_group(self, process_num, shared_dict, killed, injured):
        """Score groups of collisions by zip code"""
        # Deque is a list-like container with faster performance
        scores = deque()
        # print("I am working on " + str(len(killed)) + " collisions...")

        # Iterate between two arrays
        for x, y in zip(killed, injured):
            scores.append(self.score_row(x, y))

        # Append to destination deque
        shared_dict[process_num] = np.array(scores)
        return shared_dict

    def score_row(self, killed, injured):
        """Score individual collision row"""
        return (((killed * 2) + injured) / 5) * 5
