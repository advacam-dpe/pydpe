import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from os import path
import re
import sys
import pandas as pd

#---------------------------------
#Input variables
#---------------------------------


#---------------------------------
#Functions 
#---------------------------------

def load_clog(file_path_name, ):


	
	
#Load data from file with name and path file_in_path_name -> returns two lists with X and y positions of points
def load_file_column(file_in_path_name, index, minimum = 0, maximum = 0):
	list_column=[]
	line_num = 0


	do_range = True
	if minimum == maximum: do_range = False

	if path.exists(file_in_path_name):

		file_in = open(file_in_path_name)

		for line in file_in:

			line_num += 1
			if(line_num <= 2): continue

			line = line.replace('\n', '')
			line_list = re.split(";", line)

			if len(line_list) >= index:
				num = float(line_list[index])

				if do_range and (num > maximum or num < minimum):
					continue
			
				n += 1
				mean += num
				std += num*num
				list_column.append(num)

		file_in.close()

	else: print("File " + file_in_path_name + " does not exits.")

	return 

#Minimum and maximum value in list
def GetMinMaxListVal(List, Min = 666, Max = -666):
	if(len(List) == 0):
		return (666,-666) 

	if Min >= Max:
		Min = List[0]
		Max = List[0]

	for i in range(len(List)):
		if List[i] > Max:
			Max = List[i]
		if List[i] < Min:
			Min = List[i]
	return (Min,Max)

#Plot 1D graph with linear fuction defined with A_Slope and B_Shift
def PlotGraph1D(file_out_path_name, ListX, ListY, title, label_x, label_y):
	#Main plot function
	plt.plot(ListX,ListY, color='gainsboro', linewidth=0,
		marker='o', markerfacecolor='dodgerblue', markeredgewidth=0,
		markersize=1,label=legen_data)

     #Additional settings 
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.title(title)
	#plt.text(10, 144, r'an equation: $E=mc^2$', fontsize=15)
	(Ymin,Ymax) = GetMinMaxListVal(ListY)
	plt.ylim(ymin=Ymin-(Ymax-Ymin)*0.2, ymax = Ymax+(Ymax-Ymin)*0.4)
	#plt.legend(bbox_to_anchor=(0.01,0.9), loc="center left", borderaxespad=1, frameon=False)
	plt.legend(fontsize=10)
	plt.grid(visible = True, color ='grey',  linestyle ='-.', linewidth = 0.5,  alpha = 0.6)

	#fig = plt.gcf()
	#fig.set_size_inches(14.5, 8.5)

	#Save and close file
	plt.savefig(file_out_path_name, dpi=600)
	#plt.close()
	plt.show()

#Plto 1D histogram from data
def plot_1Dhist(elist, n_bins, x_range):
	
	if len(elist) <=- 0: return 1
	plt.hist(elist,bins=n_bins, range=x_range, histtype='step', linewidth=0.8)


	#plt.ylim(ymin=1e-4, ymax = BinContMax*3) #Set image range to nicely fit also the legend
	#plt.xlim(xmin=4, xmax = 1.2e4) #Set image range to nicely fit also the legend
	plt.grid(visible = True, color ='grey',  linestyle ='-.', linewidth = 0.5,  alpha = 0.6) #Grid on background
	#plt.xlabel(label_x, fontsize=12)  #Set label of X axis
	plt.ylabel("N [-]", fontsize=12)  #Set label of Y axis
	#plt.title(title)    #Set title
	#plt.yscale('log')
	#plt.xscale('log')


	plt.show() 
	return 0

def plot_hist1d_auto(elist, mean, std, factor = 3):
	plot_1Dhist(elist, 200, [mean - factor*std,mean + factor*std])

#---------------------------------
#Main processing part
#---------------------------------


if __name__ == '__main__':
	main()
