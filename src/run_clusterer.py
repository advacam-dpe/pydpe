import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from os import path
import re
import sys
import time
import multiprocessing
import subprocess


def run_clusterer(clusterer, file_in_path_name, file_out_path = "./", calib_dir = "", 
					clog_name = "", elist2_name = "", sens_thick = 500):
	rc = 0

	if len(clusterer) == 0 or len(file_in_path_name) == 0:
		return -1

	cmd = clusterer + " "
	if len(calib_dir) != 0:
		cmd += " -c " + calib_dir	
	if len(clog_name) != 0:
		cmd += " -l " + file_out_path + clog_name		
	if len(elist2_name) != 0:
		cmd += " --extendedevlist --evlist2 " + file_out_path + elist2_name
	cmd += " --el-thickness " + str(sens_thick)
	cmd += " " + file_in_path_name

	print(cmd)

	rc = os.system(cmd)

	return rc



if __name__ == '__main__':
	clusterer = "/mnt/MainDisk/Soubory/Programy/Vlastni/c++/aplikace/DataProcessing/PreProcessing/clusterer/out/clusterer"
	file_in_path_name = "/mnt/MainDisk/Soubory/Programy/Vlastni/python/aplikace/advacam/dpe/devel/test/run_clusterer/in/tot_toa.t3pa"
	file_out_path = "/mnt/MainDisk/Soubory/Programy/Vlastni/python/aplikace/advacam/dpe/devel/test/run_clusterer/out/"
	calib_dir = ""
	elist2_name = "Elist"
	clog_name = "ClusterLog"

	run_clusterer(clusterer, file_in_path_name, file_out_path, calib_dir, clog_name, elist2_name)