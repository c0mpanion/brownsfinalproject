#!/usr/bin/env python3

import numpy as np


class ParallelStrategy:
    """Parallel strategy approach"""
    def __init__(self, data_frame):
        print("Starting parallel strategy...")
        self.df = data_frame
        self.df_zip = self.df.groupby('ZIP CODE')

        df_zip_count = len(self.df_zip.groups);
        print("Found {} unique zip codes.".format(df_zip_count))

        # for zip in df_zipcodes.group:
        #     print()

    def print_columns(self):
        print(self.df.columns)
