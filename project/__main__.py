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
    # mod = SourceModule("""
    # __global__ void scoring(
    #
    # """)


def allocate_send_data_gpu(np_array):
    # TODO: send a numpy/cuda array to the gpu and allocate blocks within the gpu for it, used by scoring function
    """Adds a severity score column to our csv data frame"""


def add_column(data_df):
    null_data = []
    data_df['Severity Score'] = null_data
    return data_df


def import_csv():
    """Imports the NYPD collisions csv and returns a data frame
    for manipulation"""
    data_df = pd.read_csv(os.path.join(os.path.dirname(__file__)), './data/NYPD_Motor_Vehicle_Collisions.csv')
    return data_df


def main():
    # get the start time
    start_time = time.time()

    # convert csv to data frame
    data_df = import_csv()

    # tell us how long csv file took to run
    print("Reading csv file took ", time.time() - start_time, "to run")

    # print the columns of our csv file
    print(data_df.columns)

    # add column for scoring
    data_df = add_column(data_df)

    # pulling needed columns from data frame for severity score
    df_killed = data_df['Number of Persons Killed'].tolist()
    df_injured = data_df['Number of Persons Injured'].tolist()
    df_score = data_df['Severity Score'].tolist()

    # converts the column lists to cuda arrays
    cuda_killed = np.array(df_killed, dtype=np.int32)
    cuda_injured = np.array(df_injured, dtype=np.int32)
    cuda_score = np.array(df_score, dtype=np.int32)

    # score the cuda arrays using scoring function
    data_df = scoring(cuda_killed, cuda_injured, cuda_score)


main()
