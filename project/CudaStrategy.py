import time
import numpy as np
import pandas as pd

import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as cuda


class CudaStrategy:
    def __init__(self, data_frame, thread_count):
        """Cuda strategy approach"""
        if thread_count is None:
            thread_count = 512

        print("\nStarting Cuda strategy with {} threads...".format(thread_count))
        self.df = data_frame

        # Run scoring function
        start_time = time.time()
        CudaStrategy.total_scores = self.score_df(thread_count)
        print("* Cuda strategy completed in {} seconds with {} scores..."
              .format(time.time() - start_time, len(CudaStrategy.total_scores)))

        # Test for invalid data or scoring function change
        if CudaStrategy.total_scores[1000000][0] != 0 or CudaStrategy.total_scores[1000001][0] != 0.625:
            raise ValueError("Cuda returned an unexpected score, [...{}, {}...].".format(
                CudaStrategy.total_scores[1000000][0], CudaStrategy.total_scores[1000001][0]
            ))

    def get_core_size(self, thread_count, n):
        return int(round(n/thread_count + 1))

    def score_df(self, thread_count):
        """Scores each collision using a scoring function that gives
        a score of 2 to each person that was killed, a score of 1
        to each person injured, and divides those two scores added
        up by an average of 20 people per accident, then multiplies
        that fraction by 5 for a severity score of 0-5"""

        mod = SourceModule("""
        __global__ void score_function(float *dest, float *killed, float *injured)
        {
            const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
            dest[i] = (((killed[i] * 2.0) + injured[i]) / 8.0) * 5.0;
        }
        """)

        df = self.df[[
            'SCORE',
            'LATITUDE',
            'LONGITUDE',
            'NUMBER OF PERSONS KILLED',
            'NUMBER OF PERSONS INJURED'
        ]].values.astype(np.float32)

        # Calculate kernel params
        n = len(df[:, 0])
        output = np.zeros_like(df[:, 0])
        thread_size = thread_count
        core_size = self.get_core_size(thread_count, n)

        # Run kernel
        score_function = mod.get_function("score_function")
        score_function(
            cuda.Out(output),
            cuda.In(df[:, 3]),
            cuda.In(df[:, 4]),
            block=(thread_size, 1, 1),
            grid=(core_size, 1)
        )

        # Only return score with lat/long
        return df[:, [0, 1, 2]]
