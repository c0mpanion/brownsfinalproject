import numpy as np
# import pycuda.driver as cuda
# import pycuda.autoinit
# from pycuda.compiler import SourceModule
# from gmplot import gmplot


class CudaStrategy:
    """Cuda strategy approach"""
    def __init__(self, data_frame):
        print("Starting Cuda strategy...")
        self.df = data_frame

    def print_columns(self):
        print(self.df.columns)
