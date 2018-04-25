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
        total_scores = self.score_df()
        print("* Cuda strategy completed in {} seconds with {} scores..."
              .format(time.time() - start_time, len(total_scores)))

    def add_column(self):
        """ Adds severity score column filled with zeros to the data frame """
        self.df['SEVERITY SCORE'] = np.zeros

    def print_columns(self):
        """ Prints header columns of the data frame """
        print(self.df.columns)

    def score_df(self):
        """Scores each collision using a scoring function that gives
        a score of 2 to each person that was killed, a score of 1
        to each person injured, and divides those two scores added
        up by an average of 20 people per accident, then multiplies
        that fraction by 5 for a severity score of 0-5"""

        mod = SourceModule("""
        __global__ void score_function(float *dest, float* killed, float* injured)
        {
            const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
            dest[i] = ((((killed[i] * 2.0) + (injured[i] * 1.0)) / 8.0) * 5.0);
        }
        """)
        score_function = mod.get_function("score_function")

        killed = self.df['NUMBER OF PERSONS KILLED'].values.astype(np.float32)
        injured = self.df['NUMBER OF PERSONS INJURED'].values.astype(np.float32)

        dest = np.zeros_like(killed)

        # Run kernel
        score_function(
            cuda.Out(dest),
            cuda.In(killed),
            cuda.In(injured),
            block=(1024, 1, 1),
            grid=(1213, 1)
        )
        
        return dest

