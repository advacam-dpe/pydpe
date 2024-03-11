import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from os import path
import re
import sys
import pandas as pd
import hist1d as ht1d
from tqdm import tqdm

from cluster import *

class Clog(object):
    """docstring for Clog"""
    def __init__(self, file_in_path_name = "", usecols=None, nrows=None, do_print=True):
        super(Clog, self).__init__()

        self.basic_init()

        self.file_in_path_name = file_in_path_name

    def basic_init(self):
        self.file_in_path_name = ""

        self.frame_count = 0

        self.clusters = []

        self.done_load = False

    def init(self):
        pass

    def load(self, file_path_name = ""):
        if file_path_name: 
            self.file_in_path_name = file_path_name
        else:
            file_path_name = self.file_in_path_name

        try:
            with open(file_path_name, "r") as file_in:

                for idx, line in enumerate(file_in):
                    if "F" == line[0]:
                        self.frame_count += 1
                    else:
                        try:
                            cluster = Cluster()
                            cluster.load_from_string(line)
                            self.clusters.append(cluster)
                        except Exception as e:
                            print(f"error - failed to read cluster from line {idx}: {e}")

            self.done_load = True

        except IOError:
            print("error - can not open file: " + file_path_name )    


if __name__ == '__main__':
    
    case = 1 

    if case == 1:
        
        file_path_name = "devel/clog/data.clog"

        clog = Clog(file_path_name)
        clog.init()
        clog.load()

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12,10))

        matrix = np.zeros((256,256))

        for idx, cluster in enumerate(clog.clusters):
            matrix += cluster.convert_pixels_to_matrix()

        plt.imshow(matrix, origin="lower")
        plt.xlabel("X [px]")
        plt.ylabel("Y [px]")
        plt.title("conversion of clog into frame")
        plt.show()