import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from os import path
import re
import sys
import pandas as pd
from matplotlib.colors import LogNorm
import copy


sys.path.append("src")
import hist1d as ht1d
import cluster as cl
from clist import *  
# import clusterer


def remove_old_clist(clist_path_name):
    """
    removes old clist to not append to existing one
    """
    if os.path.exists(clist_path_name + ".clist"):
        os.remove(clist_path_name + ".clist")

def process_with_clusterer(cmd):
    """
    process some command with clusterer
    """
    try:
        rc = os.system(clusterer_cmd)
        if rc:
            raise RuntimeError(f"error - processing with clusterer failed: rc = {rc}")
    except Exception as e:
        raise RuntimeError(f"error - failed to process current command with clusterer: {clusterer_cmd} : {cmd}")    



if __name__ == '__main__':
    script_path = os.path.dirname(os.path.realpath(__file__))
    
    # paths and names
    data_path = os.path.join(script_path, "in")

    clusterer_path = os.path.join(script_path, "bin")
    clusterer_name = "clusterer"
    clusterer_bin = os.path.join(clusterer_path, clusterer_name)

    out_dir = os.path.join(script_path, "out")

    clist_path_name = os.path.join(out_dir, "data")


    #  create some dirs
    os.makedirs(out_dir, exist_ok=True)

    # processing with clusterer
    remove_old_clist(clist_path_name)    
    
    clusterer_cmd = clusterer_bin + f" --clust-feat-list \"{clist_path_name}\" \"{data_path}\"" 
    process_with_clusterer(clusterer_cmd)

    # load clist and go through clusterers
    clist = Clist(clist_path_name + ".clist")

    """
    DetectorID  EventID X   Y   E   T   Flags   Size    Height  BorderPixCount  Roundness   AngleAzim   Linearity   LengthProj  WidthProj   IsSensEdge  StdAlong    StdPerp Thin    Thick   CurlyThin   EpixMean    EpixStd LengthCorrStd   Length3DCorrStd AngleElev   LET Diameter    ClusterID   ClusterPixels    
    """
    for idx, row in clist.data.iterrows():
        cluster = cl.Cluster()
        cluster_str = row["ClusterPixels"]

        # load one cluster pixels - if failed then skip
        try:
            cluster.load_from_string(cluster_str)
        except:
            print(f"error - failed to load cluster {idx} from row")
            continue

        # test plot
        cluster.plot()

        # iterating tough list of pixels
        for pix in cluster.pixels:
            print(pix.x, pix.y, pix.index, pix.tot, pix.toa, pix.count)

