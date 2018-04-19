import numpy as np
import pandas as pd
import time
import os
# import pycuda.driver as cuda
# import pycuda.autoinit
# from pycuda.compiler import SourceModule
# from gmplot import gmplot


def scoring(killed, injured, score):
    # TODO: Write the severity algorithm given by Dr. C and use allocate function to send data there
    """Send the lists to the gpu, score each row, return lists back to CPU"""
    killsize = np.ndarray.size(killed)
    injuredsize = np.ndarray.size(injured)
    n = killsize + injuredsize
    numThreads = 20
    numCores = (n / 20) + 1
    mod = SourceModule("""
     __global__ void scoring(float* a, float* b, long n)
    long element = blockIdx.x * blockDim.x + threadIdx.x;
    
    """)

    return (killed * 0.5) * (injured * 0.15)


def allocate_send_data_gpu(np_array):
    # TODO: send a numpy/cuda array to the gpu and allocate blocks within the gpu for it, used by scoring function
    """Adds a severity score column to our csv data frame"""


def add_column(data_frame):
    data_frame['Severity Score'] = []
    return data_frame


def import_csv():
    """Imports the NYPD collisions csv and returns a data frame
    for manipulation"""
    data_location = (os.path.join(os.path.dirname(__file__), 'data/NYPD_Motor_Vehicle_Collisions.csv'))
    data_frame = pd.read_csv(data_location, index_col='DATE')

    return data_frame


def main():
    print("Python version: " + sys.version)
    print("Pandas version: " + pd.__version__)
    print("numpy version: " + np.__version__)

    # get the start time
    start_time = time.time()

    # convert csv to data frame
    df_collisions = import_csv()

    # tell us how long csv file took to run
    print("Reading csv file took ", time.time() - start_time, "to run")

    # print the columns of our csv file
    print(df_collisions.columns)
    # print(df_collisions.head())

    # Group data frame by zip code
    df_zipcodes = df_collisions.groupby('ZIP CODE')

    print(len(df_zipcodes.groups))

    for index, row in df_zipcodes.iterrows():
        print(row)
        # row['c1'], row['c2']

    # ['NUMBER OF PEDESTRIANS INJURED'].mean()
    # print(mean)
    # collisions.describe(include="all")

    # add column for scoring
    # data_frame = add_column(data_frame)

    # pulling needed columns from data frame for severity score
    df_killed = df_collisions['NUMBER OF PERSONS KILLED'].tolist()
    df_injured = df_collisions['NUMBER OF PEDESTRIANS INJURED'].tolist()
    # df_score = data_frame['Severity Score'].tolist()

    # converts the column lists to cuda arrays
    array_killed = np.array(df_killed, dtype=np.int32)
    array_injured = np.array(df_injured, dtype=np.int32)
    # array_score = np.array(df_score, dtype=np.int32)

    # plt.plot(array_killed)
    # plt.plot(array_injured)
    # plt.show()

    # score the cuda arrays using scoring function
    # data_frame = scoring(cuda_killed, cuda_injured, cuda_score)


main()
