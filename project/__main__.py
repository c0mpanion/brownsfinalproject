import numpy as np
import pandas as pd
import time
import os
import sys
import argparse
import gmplot

from LinearStrategy import LinearStrategy
from ParallelStrategy import ParallelStrategy
from CudaStrategy import CudaStrategy
from FastPlot import FastPlotter




def import_csv():
    """Imports the NYPD collisions csv and returns a Pandas data frame for manipulation"""
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
    """Parse arguments from the command line for thread count"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "-threads", help="specifies a thread count for parallel operations", type=int)
    return parser.parse_args()


"""Merges lat/long data with scores, then extracts scores based on ranges, sends each ranges lats/longs to gmplot 
for mapping"""


def plot_points(df):
    """Note: Our multithreaded heat map function is within the gmplot.py, however FastPlot.py is available
        to read in isolation. These functions however, do not call our multithreaded function (it calls
        the normal gmplot heat map function) because our multithreaded function doesn't work
        due to processes hanging/children processes never returning"""

    # Group all lats/longs by score in a 2D array
    latlongscores = np.column_stack((df[1], df[2], df[0]))

    # Get the lats/longs for all scores between 0-1
    scoreslessthanone = latlongscores[np.where((latlongscores[:, 2] > 0.0) * (latlongscores[:, 2] <= 1.0))]
    scoreslessthanonelats = scoreslessthanone[:, [0]]
    scoreslessthanonelongs = scoreslessthanone[:, [1]]

    # Get the lats/longs for all scores between 1-2
    scoresonetotwo = latlongscores[np.where((latlongscores[:, 2] > 1.0) * (latlongscores[:, 2] <= 2.0))]
    scoresonetotwolats = scoresonetotwo[:, [0]]
    scoresonetotwolongs = scoresonetotwo[:, [1]]

    # Get the lats/longs for all scores between 2-3
    scorestwotothree = latlongscores[np.where((latlongscores[:, 2] > 2.0) * (latlongscores[:, 2] <= 3.0))]
    scorestwotothreelats = scorestwotothree[:, [0]]
    scorestwotothreelongs = scorestwotothree[:, [1]]

    # Get the lats/longs for all scores between 3-4
    scoresthreetofour = latlongscores[np.where((latlongscores[:, 2] > 3.0) * (latlongscores[:, 2] <= 4.0))]
    scoresthreetofourlats = scoresthreetofour[:, [0]]
    scoresthreetofourlongs = scoresthreetofour[:, [1]]

    # Get the lats/longs for all scores between 4-5
    scoresfourtofive = latlongscores[np.where((latlongscores[:, 2] > 4.0) * (latlongscores[:, 2] <= 5.0))]
    scoresfourtofivelats = scoresfourtofive[:, [0]]
    scoresfourtofivelongs = scoresfourtofive[:, [1]]


    # Plot scores 0-1 on its own map using gmplot's native heatmap function
    start_time = time.time()
    gmap = gmplot.GoogleMapPlotter(40.730610, -73.935242, 20)
    gmap.heatmap(scoreslessthanonelats, scoreslessthanonelongs)
    gmap.draw("scores_less_than_one.html")
    print("Finished plotting scores less than one in {} seconds...".format(time.time() - start_time))

    # Plot scores 1-2 on its own map using gmplot's native heatmap function
    start_time = time.time()
    gmap2 = gmplot.GoogleMapPlotter(40.730610, -73.935242, 20)
    gmap2.heatmap(scoresonetotwolats, scoresonetotwolongs)
    gmap2.draw("scores_one_two.html")
    print("Finished plotting scores between one and two in {} seconds...".format(time.time() - start_time))

    # Plot scores 2-3 on its own map using gmplot's native heatmap function
    start_time = time.time()
    gmap3 = gmplot.GoogleMapPlotter(40.730610, -73.935242, 20)
    gmap3.heatmap(scorestwotothreelats, scorestwotothreelongs)
    gmap3.draw("scores_two_three.html")
    print("Finished plotting scores between two and three in {} seconds...".format(time.time() - start_time))

    # Plot scores 3-4 on its own map using gmplot's native heatmap function
    start_time = time.time()
    gmap4 = gmplot.GoogleMapPlotter(40.730610, -73.935242, 20)
    gmap4.heatmap(scoresthreetofourlats, scoresthreetofourlongs)
    gmap4.draw("scores_three_four.html")
    print("Finished plotting scores between three and four in {} seconds...".format(time.time() - start_time))

    # Plot scores 4-5 on its own map using gmplot's native heatmap function
    start_time = time.time()
    gmap5 = gmplot.GoogleMapPlotter(40.730610, -73.935242, 20)
    gmap5.heatmap(scoresfourtofivelats, scoresfourtofivelongs)
    gmap5.draw("scores_four_five.html")
    print("Finished plotting scores between four and five in {} seconds...".format(time.time() - start_time))


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
    s1 = LinearStrategy(df_collisions)
    s2 = ParallelStrategy(df_collisions, args.t)
    s3 = CudaStrategy(df_collisions, args.t)

    print("\nPlotting points...")

    # Plot on graph
    plot_points(s3.get_scores())


main()
