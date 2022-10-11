import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from os import path
import re
import sys

#---------------------------------
#Input
#---------------------------------

file_hist_path = "../devel/in/hist/"
file_hist_base_name = "Hist1D_E_EqBin_Total"

#---------------------------------
#Functions 
#---------------------------------

#Load data from file with name and path file_in_pathName -> returns two lists with X and y positions of points
def load_hist(file_in_path, file_in_base_name, fill_sparse_export=True, file_in_end_data = ".hist", file_in_end_info = ".hist_info"):

     bin_conts=[]
     low_edges=[]
     
     if path.exists(file_in_path + file_in_base_name + file_in_end_data):
          file_in = open(file_in_path + file_in_base_name + file_in_end_data)
          line_num = 0
          bin_width = 0
          curr_low_edge = 0
          pre_low_edge = 0

          for line in file_in:
               line_num += 1

               line = line.replace('\n', '')
               line = line.replace('\r', '') 
               if(len(line) == 0): continue
               line_list = re.split("\t", line)

               if(len(line_list)<2): continue

               curr_low_edge = float(line_list[0])
               curr_bin_conts = float(line_list[1])

               if line_num == 2: bin_width = curr_low_edge - pre_low_edge

               #If there is a gap in the data -> fill it with 0
               if(bin_width != 0 and fill_sparse_export):
                    while (curr_low_edge - pre_low_edge)/bin_width > 1:
                         pre_low_edge += bin_width
                         bin_conts.append(0)
                         low_edges.append(pre_low_edge)

               bin_conts.append(curr_bin_conts)
               low_edges.append(curr_low_edge)

               pre_low_edge = curr_low_edge

          file_in.close()

     else: 
          print("Could not open data file!", file_in_path + file_in_base_name)
          return ([-1],[-1])    

     return (low_edges, bin_conts)

#Plot hist 1D and save it
def plot_hist1d(low_edges,bin_conts, legend_data="", color="C0"):

     plt.hist(low_edges, low_edges, weights=bin_conts, color=color, histtype='step',label=legend_data, linewidth=0.8)


#---------------------------------
#Main processing part
#---------------------------------

if __name__ == '__main__':

     bin_conts, low_edges = load_hist(file_hist_path, file_hist_base_name, True)
     plot_hist1d(bin_conts, low_edges, "data", "C0")

     # plt.ylim(ymin=1, ymax = 2) #Set image range to nicely fit also the legend
     plt.xlim(xmin=-1e2, xmax = 1.5e4) #Set image range to nicely fit also the legend
     plt.legend(title="Legend") #Add legend into plot
     plt.grid(visible = True, color ='grey',  linestyle ='-.', linewidth = 0.5,  alpha = 0.6) #Grid on background
     plt.xlabel("X", fontsize=12)  #Set label of X axis
     plt.ylabel("Y", fontsize=12)  #Set label of Y axis
     plt.title("Title")    #Set title
     plt.yscale('log')
     # plt.xscale('log') 

     plt.show()
