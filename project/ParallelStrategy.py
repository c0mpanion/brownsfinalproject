import multiprocessing
import time
from itertools import chain

import numpy as np


class ParallelStrategy:
    def __init__(self, data_frame, thread_count):
        """Parallel strategy approach using threads"""
        if thread_count is None:
            thread_count = 32

        print("\nStarting parallel strategy with {} threads...".format(thread_count))
        self.df = data_frame

        # Run scoring function in parallel
        start_time = time.time()
        scores = self.score_df(thread_count)
        print("* Parallel strategy completed in {} seconds with {} scores..."
              .format(time.time() - start_time, len(scores)))

        # Test for invalid data or scoring function change
        if scores[1000000][0] != 0 or scores[1000001][0] != 0.625:
            raise ValueError("Parallel strategy returned an unexpected score, [...{}, {}...].".format(
                scores[1000000][0], scores[1000001][0]
            ))

    def score_df(self, thread_count):
        """Score a data frame in parallel with a thread count"""
        df = self.df[[
            'SCORE',
            'LATITUDE',
            'LONGITUDE',
            'NUMBER OF PERSONS KILLED',
            'NUMBER OF PERSONS INJURED',
        ]].values.astype(np.float)

        # Split array into equal number of elements for each thread to handle
        chunks = np.array_split(df, thread_count)

        # Shared dictionary for each thread to communicate (see: "pickling")
        manager = multiprocessing.Manager()
        shared_dict = manager.dict()

        # List of jobs
        jobs = []

        # Create a thread to handle each chunk (split) of items
        for i in range(thread_count):
            p = multiprocessing.Process(target=self.score_group, args=(i, shared_dict, chunks[i]))
            jobs.append(p)
            p.start()

        # Wait for all jobs to finish
        for proc in jobs:
            proc.join()

        # Merge dictionary values into array
        values = chain.from_iterable(shared_dict.values())
        return np.array(list(values))[:, [0, 1, 2]]

    def score_group(self, process_num, shared_dict, chunk):
        """Score groups of collisions by group of elements"""
        for i in range(len(chunk)):
            chunk[i][0] = self.score_row(chunk[i][3], chunk[i][4])

        # Add array to shared memory
        shared_dict[process_num] = chunk

    def score_row(self, killed, injured):
        """Score individual collision row"""
        return (((killed * 2.0) + injured) / 8.0) * 5.0
