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

rejection_const = 	5000
file_in_path_name = "/mnt/MainDisk/Soubory/Programy/Vlastni/c++/aplikace/DataProcessing/Processing/DPE/Devel/Test/Histograms/2D/ClusterSensorPlots/ClusterPlotEnergy_Integrated.txt"
file_out_path = 	"../devel/out/mask/"
file_out_name = 	"Mask.txt"

#===========================================================================
#Functions
#===========================================================================

def load_matrix(file_in_path_name):

	matrix = np.zeros((256, 256))	
	line_num = 0

	if path.exists(file_in_path_name):
		file_in = open(file_in_path_name)
		for line in file_in:

			line_list = re.split("\t", line)

			if len(line_list) < 5: 
				continue
			
			line_list = line_list[0:256]

			for i in range(len(line_list)):
				if i >= 256: print("i ", i)
				if line_num >= 256: print("line_num ", line_num)

				matrix[line_num, i] = line_list[i]

			line_num += 1

		file_in.close()

	return matrix

def mask_matrix(matrix, rejection_const):
	
	matrix_mask = []

	for i in range(len(matrix)):
		matrix_mask.append([])

		for j in range(len(matrix[0])):
			
			if(matrix[i,j] > rejection_const): 	
				print(i,j)
				matrix_mask[i].append(0)
			else:
				matrix_mask[i].append(1)				

	return matrix_mask

def plot_matrix(matrix, file_out_path_name):
	fig = plt.figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
	plt.imshow(matrix, cmap = 'cividis', aspect = 'auto', origin='lower')
	#plt.show()
	plt.savefig(file_out_path_name)
	plt.close()

def export_mask(pix_mask_list, file_out_path_name):
	file = open(file_out_path_name,"w")

	if not path.exists(file_in_path_name):
		print("Can not export mask to:\t", file_out_path_name)
		return -1

	for i in range(len(pix_mask_list)):
		for j in range(len(pix_mask_list[0])):
			file.write(str(pix_mask_list[i][j]) + " ")
		file.write("\n")

	file.close()

#===========================================================================
#Main processing part
#===========================================================================

if __name__ == '__main__':

	matrix = load_matrix(file_in_path_name)
	plot_matrix(matrix, file_out_path + "matrix_integ_no_mask.png")
	matrix_mask = mask_matrix(matrix, rejection_const)
	print(matrix_mask)
	# plot_matrix(matrix_mask, file_out_path + "matrix_integ_mask.png")
	# export_mask(matrix_mask, file_out_path + file_out_name)