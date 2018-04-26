import time
import numpy as np
import pandas as pd

import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as cuda


class CudaStrategy:
    """Cuda strategy approach"""
    def __init__(self, data_frame):
        print("\nStarting Cuda strategy...")
        self.df = data_frame

        # Run scoring function
        start_time = time.time()
        CudaStrategy.total_scores = self.score_df()
        print("* Cuda strategy completed in {} seconds with {} scores..."
              .format(time.time() - start_time, len(CudaStrategy.total_scores)))

        # Test for invalid data or scoring function change
        if CudaStrategy.total_scores[1000000] != 0 or CudaStrategy.total_scores[1000001] != 0.625:
            raise ValueError("Cuda returned an unexpected score, [...{}, {}...].".format(
                CudaStrategy.total_scores[1000000], CudaStrategy.total_scores[1000001]
            ))


    def add_column(self):
        """ Adds severity score column filled with zeros to the data frame """
        self.df['SEVERITY SCORE'] = np.zeros

    def print_columns(self):
        """ Prints header columns of the data frame """
        print(self.df.columns)

    def get_thread_size(self):
        return 512

    def get_core_size(self, n):
        return int(round(n/self.get_thread_size() + 1))

    def score_df(self):
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
        killed = self.df['NUMBER OF PERSONS KILLED'].values.astype(np.float32)
        injured = self.df['NUMBER OF PERSONS INJURED'].values.astype(np.float32)

        # Calculate kernel params
        n = len(killed)
        output = np.zeros_like(killed)
        thread_size = self.get_thread_size()
        core_size = self.get_core_size(len(output))

        # Run kernel
        score_function = mod.get_function("score_function")

        score_function(
            cuda.Out(output),
            cuda.In(killed),
            cuda.In(injured),
            block=(thread_size, 1, 1),
            grid=(core_size, 1)
        )

        return output

