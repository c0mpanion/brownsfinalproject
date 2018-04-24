import numpy as np
import pandas as pd
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
# from gmplot import gmplot
from pycuda.elementwise import ElementwiseKernel


class CudaStrategy:
    """Cuda strategy approach"""

    def __init__(self, data_frame):
        print("Starting Cuda strategy...")
        self.df = data_frame
        self.add_column(self.df)

        # Convert pandas data frame columns to numpy lists
        persons_killed = self.df[['NUMBER OF PERSONS KILLED']]
        persons_injured = self.df[['NUMBER OF PERSONS INJURED']]
        severity_score = self.df[['SEVERITY SCORE']]

        persons_killed[1] = 0
        persons_injured[1] = 0
        severity_score[1] = 0

        persons_killed.astype(np.int32)
        persons_injured.astype(np.int32)
        severity_score.astype(np.int32)

        print(persons_killed)
        print(persons_injured)
        print(severity_score)

        self.scoring(persons_killed, persons_injured, severity_score)

        # Convert np types as int 32 for compatibility with GPU
        # persons_killed.dtype("int32")
        # persons_injured.dtype("int32")
        # severity_score.dtype("int32")

    """ Adds severity score column filled with zeros to the data frame """
    def add_column(self, df):
        df['SEVERITY SCORE'] = np.zeros

    """ Prints header columns of the data frame """
    def print_columns(self):
        print(self.df.columns)

    """ Scores each collision using a scoring function that gives 
        a score of 2 to each person that was killed, a score of 1
        to each person injured, and divides those two scores added
        up by an average of 20 people per accident, then multiplies
        that fraction by 5 for a severity score of 0-5
    """
    def scoring(self, killed, injured, score):
        """Send the lists to the gpu, score each row, return lists back to CPU"""

        killed.astype(int)
        injured.astype(int)
        score.astype(int)

        scorefunc = SourceModule("""
        __global__ void scoring(int *killed, int *injured, int *score)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            score[i] = ((((killed[i] * 2) + (injured[i])) / 20) * 5) ;
        }
        """)

        scoring_func = scorefunc.get_function("scoring")

        scoring_func(cuda.Out(score), cuda.In(killed), cuda.In(injured), block=(400, 1, 1))

        print(score)
