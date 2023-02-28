import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from os import path
import re
import sys

#---------------------------------
#Input variables
#---------------------------------

file_in_path_name = "./output/p_70MeV_60deg/extelist.txt"             #File with input data
file_out_path_name = "./output/p_70MeV_60deg/hist1D_LET.png"        #File for output image

label_x = "Count [-]"				#Label on X axis
label_y = "Time [ns]"					#Label on Y axis
title = "Time evolution"     			#title of the graph
legen_data = "Time from elist"			#Label in the legend for the data

sens_thick = 	500
part_dir = 		60
E_dep_mean = 	1960

#---------------------------------
#Functions 
#---------------------------------

#Load data from file with name and path file_in_path_name -> returns two lists with X and y positions of points
def load_file_column(file_in_path_name, index, minimum = 0, maximum = 0):
	list_column=[]
	line_num = 0

	mean = 0
	std = 0
	n = 0

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

	mean = mean/float(n)

	if n > 1:
		std = math.sqrt((std - mean*mean*float(n))/float(n - 1));
	else:
		std = 0

	print("--------------------")
	print("Column:\t", index)
	print("N:\t", n)
	print("Mean:\t", mean)
	print("Std:\t", std)

	return (list_column, mean, std)

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

# (ListComulmTime) = load_file_column(file_in_path_name,5)
# (ListComulmE) = load_file_column(file_in_path_name,4)
# (ListComulmX) = load_file_column(file_in_path_name,2)

# print(GetMinMaxListVal(ListComulmE))

# ListX = []

# FirstTime = ListComulmTime[0]

# for x in range(len(ListComulmTime)):
# 	ListX.append(x);
# 	ListComulmTime[x] -= FirstTime
# 	print(x,"\t",ListComulmTime[x])

# PlotGraph1D(file_out_path_name, ListX,ListComulmTime, title, label_x, label_y);

list_LET, LET_mean, LET_std = load_file_column(file_in_path_name, 26, 1.3, 3.5)
list_E, E_mean, E_std = load_file_column(file_in_path_name, 4, 1300, 3000)
list_L3D, L3D_mean, L3D_std = load_file_column(file_in_path_name,24, 800, 1200)
list_L2D_corr, L2D_corr_mean, L2D_corr_std = load_file_column(file_in_path_name,23, 10, 20)

# plot_hist1d_auto(list_LET, LET_mean, LET_std)
# plot_1Dhist(list_E, 200, [E_mean - 3*E_std,E_mean + 3*E_std])
# plot_1Dhist(list_L3D, 200, [L3D_mean - 3*L3D_std,L3D_mean + 3*L3D_std])
# plot_1Dhist(list_L2D_corr, 200, [L2D_corr_mean - 3*L2D_corr_std,L2D_corr_mean + 3*L2D_corr_std])

L2D_est = (sens_thick/55.)/math.tan( (90-part_dir)*math.pi/180.)
L3D_est = math.sqrt(L2D_est*L2D_est*55.*55. + sens_thick*sens_thick) 

print("============================================")
print("Sensor thickness:\t" , sens_thick , "um")
print("Particle direction:\t" , part_dir , "deg")
print("Lenght 2D estimated:\t" , L2D_est , "px")
print("Lenght 3D estimated:\t" , L3D_est , "um")
print("LET estimated:\t\t" , E_dep_mean/L3D_est , "keV/um")
print("============================================")
print("Lenght 2D estimated:\t" , L2D_est , "px")
print("Lenght 2D from clust:\t" , L2D_corr_mean , "px")
print("============================================")
print("Mean LET form E_mean/L3D_mean:\t", E_mean/L3D_mean)
print("Mean LET form E_mean/pythL2D:\t", E_mean/math.sqrt(L2D_corr_mean*L2D_corr_mean*55.*55. + sens_thick*sens_thick))
print("Mean LET from clusterer:\t", LET_mean)
print("Mean LET from estimated:\t", E_dep_mean/L3D_est)
print("============================================")



plot_hist1d_auto( *load_file_column(file_in_path_name, 22) )
plot_hist1d_auto( *load_file_column(file_in_path_name, 21) )
plot_hist1d_auto( *load_file_column(file_in_path_name, 20) )
plot_hist1d_auto( *load_file_column(file_in_path_name, 19) )
plot_hist1d_auto( *load_file_column(file_in_path_name, 18) )
plot_hist1d_auto( *load_file_column(file_in_path_name, 17) )
plot_hist1d_auto( *load_file_column(file_in_path_name, 16) )
plot_hist1d_auto( *load_file_column(file_in_path_name, 15) )

