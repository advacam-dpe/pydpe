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

FileIn_PathName = "/mnt/MainDisk/Soubory/Analysis/TGRKiyoshi/data/Pixels_92_185_1671339280.t3pa"             #File with input data
FileOut_Path = "/mnt/MainDisk/Soubory/Analysis/Stack/out/"        #File for output image

#---------------------------------
#Functions 
#---------------------------------

#Load data from file with name and path FileIn_PathName -> returns two lists with X and y positions of points
def LoadFileColumn(FileIn_PathName, Index=0):
	ListColumn=[]
	LineNum = 0
	if path.exists(FileIn_PathName):
		FileIn = open(FileIn_PathName)
		for Line in FileIn:
			LineNum += 1
			if(LineNum == 1): continue
			Line = Line.replace('\n', '')
			LineList = re.split(" |\t|;", Line)
			if len(LineList) >= Index:
				ListColumn.append(float(LineList[Index]))
				#print(Line)
		FileIn.close()
	return (ListColumn)

#Load data from file with name and path FileIn_PathName 
def InspectFileColumn(FileIn_PathName, Index=0):

	Count = 0
	Min = 1e300
	Max = -1e300
	Mean = 0
	MaxPosDif = -1e300
	MaxNegDif = -1e300

	Prev = -666.666

	LineNum = 0
	if path.exists(FileIn_PathName):
		FileIn = open(FileIn_PathName)
		for Line in FileIn:
			LineNum += 1
			if(LineNum == 1): continue
			Line = Line.replace('\n', '')
			LineList = re.split(" |\t|;", Line)
			if len(LineList) >= Index:
				Num = float(LineList[Index])
				Count+= 1
				Mean += Num
				if(Num > Max): 
					Max = Num
					print(Line)
				if(Num < Min): Min = Num
				if(Prev != -666.666 and MaxPosDif < (Num - Prev)): MaxPosDif = Num - Prev
				if(Prev != -666.666 and MaxNegDif < (Prev - Num)): MaxNegDif = Prev - Num
				Prev = Num
		FileIn.close()

		Mean /= Count

		print("Mean\t", Mean )
		print("Min\t", Min )
		print("Max\t", Max )
		print("MaxPosDif\t", MaxPosDif )		
		print("MaxNegDif\t", MaxNegDif )
	
	return (Min,Max,Mean,MaxNegDif,MaxPosDif)

#Load data from file with name and path FileIn_PathName -> returns two lists with X and y positions of points
def LoadFile(FileIn_PathName):
	ListToT=[]
	ListToA=[]	
	LineNum = 0
	if path.exists(FileIn_PathName):
		FileIn = open(FileIn_PathName)
		for Line in FileIn:
			LineNum += 1
			# if(LineNum >= 1e6): break			
			if(LineNum == 1): continue
			Line = Line.replace('\n', '')
			LineList = re.split(" |\t|;", Line)
			if len(LineList) >= 4:
				ListToA.append(float(LineList[2]))
				ListToT.append(float(LineList[3]))				
				#print(Line)
		FileIn.close()
	return (ListToA, ListToT)

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
def PlotGraph1D(FileOut_PathName, ListX, ListY, Title, LabelX, LabelY, Legend):
	#Main plot function
	plt.plot(ListX,ListY, color='gainsboro', linewidth=0,
		marker='o', markerfacecolor='dodgerblue', markeredgewidth=0,
		markersize=1, label=Legend)
    #Additional settings 
	plt.xlabel(LabelX)
	plt.ylabel(LabelY)
	plt.title(Title)
	#plt.text(10, 144, r'an equation: $E=mc^2$', fontsize=15)
	(Ymin,Ymax) = GetMinMaxListVal(ListY)
	plt.ylim(ymin=Ymin-(Ymax-Ymin)*0.2, ymax = Ymax+(Ymax-Ymin)*0.4)
	#plt.legend(bbox_to_anchor=(0.01,0.9), loc="center left", borderaxespad=1, frameon=False)
	plt.legend(fontsize=10)
	plt.grid(visible = True, color ='grey',  linestyle ='-.', linewidth = 0.5,  alpha = 0.6)
	#Save and close file

	plt.savefig(FileOut_PathName, dpi=600)
	#plt.close()
	plt.show()

#Plot hist 1D from list
def PlotHist1DFromList(ListEvent,NBin, Xmin, Xmax,FileOut_PathName,Title, LabelX, LabelY, Legend, Color):
	plt.hist(ListEvent, bins=NBin, range = [Xmin,Xmax], color=Color, histtype='step',label=Legend, linewidth=0.8)
	#plt.ylim(ymin=0, ymax = BinContMax*1.5) #Set image range to nicely fit also the legend
	plt.grid(visible = True, color ='grey',  linestyle ='-.', linewidth = 0.5,  alpha = 0.6) #Grid on background
	plt.xlabel(LabelX,fontsize=12)  #Set label of X axis
	plt.ylabel(LabelY,fontsize=12)  #Set label of Y axis
	plt.legend(fontsize=10)	
	plt.title(Title)    #Set title

	plt.savefig(FileOut_PathName, dpi=600) #Save plot into FileOut_PathName
	#plt.close()        #Close plot
	plt.show() 

def converttio_matrix():



def plot_matrix(matrix, file_out_path_name, do_log_z = False, names = [], values = []):
	
	# fig, ax = plt.subplots()
	fig = plt.figure(num=None, figsize=(6, 6), dpi=300, facecolor='w', edgecolor='k')
	ax = fig.add_subplot(111)
	if do_log_z:
		plt.imshow(matrix, cmap = 'viridis', aspect = 'auto', origin='lower', norm=LogNorm())
	else:
		plt.imshow(matrix, cmap = 'viridis', aspect = 'auto', origin='lower')		
	plt.colorbar()
	plt.axis('square')

	if len(names) != 0:
		props = dict(boxstyle='round', facecolor='white', alpha=0.7, linewidth=0 )
		plt.subplots_adjust(right=0.65)
		fig = plt.gcf()
		fig.set_size_inches(7.5, 4.4)

		x_par_step = 1./30.;
		if len(names) > 30:
			x_par_step = 1./float(len(names))
		i = 0

		name_width = max(len(name) for name in names)
		value_width = 1  # adjust as needed
		# text_str = '\n'.join([f'{key:<{name_width}} = {value:>{value_width}.2f}' for key, value in zip(names, values)])
		text_str = ""
		text_str = '\n'.join([f'{key:<{name_width}} = {value:>{value_width}}' for key, value in zip(names, values)])

		plt.text(1.4, 0.5 , text_str, multialignment='left', transform=ax.transAxes, 
				alpha=0.7, bbox=props, fontsize=7, family='DejaVu Sans Mono')



	plt.show()
	# plt.savefig(file_out_path_name)
	# plt.close()




#---------------------------------
#Main processing part
#---------------------------------

#Load file

(ListColumnToA,ListColumnToT) = LoadFile(FileIn_PathName)

InspectFileColumn(FileIn_PathName, 5)


#Plot ToA and time

# ListX = []
# for x in range(len(ListColumnToA)):
# 	ListX.append(x)
# PlotGraph1D(FileOut_Path + "Graph_ToA.png", ListX,ListColumnToA, "Series of ToAs", "Count [-]", "ToA [-]", "ToA from t3pa");

# ListColumnTime = []
# for x in ListColumnToA: 
# 	ListColumnTime.append(x*25)
# PlotGraph1D(FileOut_Path + "Graph_Time.png", ListX,ListColumnTime, "Series of time", "Count [-]", "Time [ns]", "Time from t3pa: ToA*25");


# Plot several histogram with tot

# PlotHist1DFromList(ListColumnToT,1022,0,1022,FileOut_Path + "Hist1D_ToT.png", "ToT distribution", "ToT [-]", "Count [-]", "ToT of pixels","C0")
