import numpy as np
import pandas as pd
import pycuda.driver as cuda
import pycuda.autoinit
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
        persons_killed = self.df[['NUMBER OF PERSONS KILLED']].values
        persons_injured = self.df[['NUMBER OF PERSONS INJURED']].values
        severity_score = self.df[['SEVERITY SCORE']].values

       # print(persons_killed)
       # print(persons_injured)

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
        killed_gpu = gpuarray.to_gpu(killed)
        injured_gpu = gpuarray.to_gpu(injured)
        scored_gpu = gpuarray.empty_like(killed_gpu)

        """Uses elementwise kernel within PyCuda: first line takes our arrays as arguments,
        the second line takes the operation on each element with i being the index, and the
        third line names the function"""
        scoring_function = ElementwiseKernel(
                "int *killed, int *injured, int *score",
                "score[i] = (((killed[i] * 2) + (injured[i])) / 2 * 20) * 5",
                "scoring_function")

        # Implements the scoring function on our GPU arrays, retrieves our new score as a np array
        scoring_function(killed_gpu, injured_gpu, scored_gpu)
        new_score = scored_gpu.get()
        print(new_score)
