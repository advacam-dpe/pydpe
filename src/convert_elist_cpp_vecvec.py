import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from os import path
import re
import sys

file_in_path_name="/mnt/MainDisk/Soubory/Programy/Vlastni/c++/aplikace/DataProcessing/Processing/DPE/Devel/API_CPP/test_app/in/elist.exelist"

file_in=open( file_in_path_name , 'r')

line_num=0

for line in file_in:
	line_num += 1

	if line_num < 3:
		continue

	line = line.replace('\t',',')
	line = line.replace('\n','')

	print("Clusters.push_back(std::vector<double>{" + line + "});")