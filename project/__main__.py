import numpy as np
import pandas as pd
import time
import os
import sys
import argparse

from LinearStrategy import LinearStrategy
from ParallelStrategy import ParallelStrategy
from CudaStrategy import CudaStrategy


def import_csv():
    """Imports the NYPD collisions csv and returns a data frame for manipulation"""
    data_location = (os.path.join(os.path.dirname(__file__), 'data/NYPD_Motor_Vehicle_Collisions.csv'))

    data_frame = pd.read_csv(data_location, dtype={
        "NUMBER OF PERSONS INJURED": int,
        "NUMBER OF PERSONS KILLED": int,
        "BOROUGH": str,
        "ZIP CODE": str,
        "LATITUDE": float,
        "LONGITUDE": float,
    })

    # Add score column set to 0.0
    data_frame['SCORE'] = pd.Series(0.0, index=data_frame.index)

    return data_frame


def parse_args():
    """Parse arguments from the command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "-threads", help="specifies a thread count for parallel operations", type=int)
    return parser.parse_args()


def main():
    print("Python version: " + sys.version)
    print("Pandas version: " + pd.__version__)
    print("numpy version: " + np.__version__)

    # Parse command line arguments
    args = parse_args()

    # get the start time
    start_time = time.time()

    # convert csv to data frame
    df_collisions = import_csv()

    # tell us how long csv file took to run
    print("Reading CSV completed in {} seconds...".format(time.time() - start_time))

    # Run strategies
    ls = LinearStrategy(df_collisions)
    ps = ParallelStrategy(df_collisions, args.t)
    cs = CudaStrategy(df_collisions)


main()
