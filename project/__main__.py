import numpy as np
import pandas as pd
import time
import os
import sys
import argparse

from LinearStrategy import LinearStrategy
from ParallelStrategy import ParallelStrategy
from CudaStrategy import CudaStrategy
from FastPlot import FastPlotter


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
    #ls = LinearStrategy(df_collisions)
    # ps = ParallelStrategy(df_collisions)
    cs = CudaStrategy(df_collisions)
    scores = CudaStrategy.total_scores

    # Pull out lat and long columns
    lats = df_collisions['LATITUDE'].values.astype(np.float32)
    longs = df_collisions['LONGITUDE'].values.astype(np.float32)

    print(np.shape(lats))
    print(np.shape(longs))
    print(np.shape(scores))


    latlongscores = np.column_stack((lats, longs, scores))
    #print(latlongscores)

    #latlongscores = np.sort(latlongscores, kind='mergesort')

    scoreslessthanone = latlongscores[np.where((latlongscores[:, 2] > 0.0) * (latlongscores[:, 2] <= 1.0))]
    print("0-1")
    print(scoreslessthanone)
    scoreslessthanonelats = scoreslessthanone[:, [0]]
    scoreslessthanonelongs = scoreslessthanone[:, [1]]


    scoresonetotwo = latlongscores[np.where((latlongscores[:, 2] > 1.0) * (latlongscores[:, 2] <= 2.0))]
    print("1-2")
    print(scoresonetotwo)
    scoresonetotwolats = scoresonetotwo[:, [0]]
    scoresonetotwolongs = scoresonetotwo[:, [1]]

    scorestwotothree = latlongscores[np.where((latlongscores[:, 2] > 2.0) * (latlongscores[:, 2] <= 3.0))]
    print("2-3")
    print(scorestwotothree)
    scorestwotothreelats = scorestwotothree[:, [0]]
    scorestwotothreelongs = scorestwotothree[:, [1]]

    scoresthreetofour = latlongscores[np.where((latlongscores[:, 2] > 3.0) * (latlongscores[:, 2] <= 4.0))]
    print("3-4")
    print(scoresthreetofour)
    scoresthreetofourlats = scoresthreetofour[:, [0]]
    scoresthreetofourlongs = scoresthreetofour[:, [1]]

    scoresfourtofive = latlongscores[np.where((latlongscores[:, 2] > 4.0) * (latlongscores[:, 2] <= 5.0))]
    print("4-5")
    print(scoresfourtofive)
    scoresfourtofivelats = scoresfourtofive[:, [0]]
    scoresfourtofivelongs = scoresfourtofive[:, [1]]


    FastPlotter.threadedHeatMap(scoreslessthanonelats, scoreslessthanonelongs)
    FastPlotter.threadedHeatMap(scoreslessthanonelats, scoreslessthanonelongs)
    FastPlotter.threadedHeatMap(scoresonetotwolats, scoresonetotwolongs)
    FastPlotter.threadedHeatMap(scorestwotothreelats, scorestwotothreelongs)
    FastPlotter.threadedHeatMap(scoresthreetofourlats, scoresthreetofourlongs)
    FastPlotter.threadedHeatMap(scoresfourtofivelats, scoresfourtofivelongs)


main()
