import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import re
import datetime
import random
import os
from os import path
import sys

#===========================================================================
#Variables
#===========================================================================

file_in_path_name = "/mnt/MainDisk/Soubory/Programy/Vlastni/python/aplikace/advacam/dpe/devel/in/extract_names/names_cpp.cpp"
file_out_path = 	"/mnt/MainDisk/Soubory/Programy/Vlastni/python/aplikace/advacam/dpe/devel/out/extract_names/"
file_out_name = 	"names.txt"

#===========================================================================
#Functions
#===========================================================================

def extract_names(file_in_path_name):

	list_names = []

	if path.exists(file_in_path_name):
		file_in = open(file_in_path_name)
		for line in file_in:

			line_list = re.split("\"", line)

			if len(line_list) < 2: 
				continue
			

			n_names = int(len(line_list)/2)

			# print(line_list, "|", len(line_list), "|", n_names)
			
			for i in range(n_names):

				if(line.find("v_Nums[") != -1):
					list_names.append(line_list[2*i + 1] + " = 1")
				if(line.find("v_Words[") != -1): 
					list_names.append(line_list[2*i + 1] + " = true")
				if(line.find("v_WordsInParen[") != -1): 
					list_names.append( line_list[2*i + 1] + " = \"hop\"")

		file_in.close()

	return list_names

def export_names(file_out_path_name, list_names):
	file_out = open(file_out_path_name, "w")

	for item in list_names:
		# print(item)
		file_out.write(item + "\n")

	file_out.close()

#===========================================================================
#Main processing part
#===========================================================================

if __name__ == '__main__':

	list_names = extract_names(file_in_path_name)
	export_names(file_out_path + file_out_name, list_names)