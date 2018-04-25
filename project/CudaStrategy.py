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

        # Run scoring function
        total_scores = self.score_df()
        print total_scores.tolist()[0:99]
        #print(total_scores.tolist())

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
        __global__ void score_function(float *dest, int* killed, int* injured)
        {
            //const int i = threadIdx.x;
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            //dest[i] = 4;
            dest[i] = ((((killed[i] * 2.0) + (injured[i] * 1.0)) / 5.0) * 5.0) ;
        }
        """)
        score_function = mod.get_function("score_function")

        killed = self.df['NUMBER OF PERSONS KILLED'].values.astype(np.int32)
        injured = self.df['NUMBER OF PERSONS INJURED'].values.astype(np.int32)

        # killed = np.array([2, 3, 4, 5])
        # injured = np.array([3, 5, 6, 1])

        print("Killed size: " + str(len(killed)))
        print("Injured size: " + str(len(injured)))

        dest = np.zeros_like(killed, dtype=(float))

        # Run kernel
        score_function(
            cuda.Out(dest),
            #cuda.In(np.int32(killed)),
            #cuda.In(np.int32(injured)),
            cuda.In(killed),
            cuda.In(injured),
            block=(1000, 1, 1),
            grid=(2,1)
        )
        # score_function(
        #     drv.Out(dest),
        #     drv.In(killed),
        #     drv.In(injured),
        #     block=(400, 1, 1)
        # )

        return dest

