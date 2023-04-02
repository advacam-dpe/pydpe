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

class elist(object):
	"""docstring for elist"""
	def __init__(self, filein_path_name = ""):
		super(elist, self).__init__()

		rc = 0

		self.basic_init()

		if len(filein_path_name) != 0:
			rc = self.load(filein_path_name)

		if rc:
			print("Error occurred during initialization.")

	def basic_init(self):
		self.filein_path_name = ""
		
		self.data = pd.DataFrame()
		self.var_keys = []
		self.var_units = []
		self.ncol = 0
		self.separator = ""
		self.nrow = 0

		self.dsc = ""


	def load(self, file_path_name):
		try:
			with open(file_path_name, 'r') as file_in:
				n_line = 0
				for line in file_in:
					n_line += 1
					if n_line == 1:
						self.__find_separator(line)
					if n_line == 2:
						self.__load_units(line)
					if n_line == 10:
						break

				if len(self.separator) != 0:
					self.data = pd.read_csv(file_path_name, sep=self.separator, header=[0], skiprows = [1])
					self.filein_path_name = file_path_name
					self.var_keys = self.data.keys() 
					self.ncol = len(self.var_keys)
					self.nrow = len(self.data)
				else:
					print("Load of data failed from file:" + file_path_name)

		except IOError:
			print("Can not open file: " + file_path_name )	
			return -1

		return 0

	def __find_separator(self, line):
		separator = ""
		if line.find("\t") != -1: separator = "\t"
		elif line.find(" ")!= -1: separator = " "
		elif line.find(";")!= -1: separator = ";"		

		if len(separator) != 0:
			self.separator = separator
		return separator

	def __load_units(self, line):
		line = line.replace("\n", "")
		self.var_units = line.split(self.separator)

	def print(self):
		print("Elist")
		if len(filein_path_name): print("File:", self.filein_path_name)
		print("VarKeys:", self.var_keys)
		print("VarUnits:", self.var_units)
		print(self.data) 

	def plot(self, var_key, nbin = 100, do_show = True, ax=None):
		max_val = np.max(self.data[var_key])
		min_val = np.min(self.data[var_key])
		range_val = max_val - min_val

		if min_val == max_val:
			range_val = abs(max_val)
			if min_val == 0:
				range_val = 10

		hist1d_1 = ht1d.hist1d(nbin = nbin, xmin = min_val - range_val*0.1, xmax = max_val + range_val*0.1)

		hist1d_1.fill_np(self.data[var_key])
		hist1d_1.title = "Histogram of " + var_key
		hist1d_1.name = var_key		
		hist1d_1.axis_labels = [var_key, "N"]
		hist1d_1.plot(do_show=do_show, ax=ax)

	def plot_all(self, nbin = 100, do_show = True):

		dim_x = int(math.sqrt(float(self.ncol)))
		dim_y = int(dim_x + 1)

		fig, axs = plt.subplots(dim_y, dim_x, sharex=False)

		for i in range(self.ncol):
			i_y = int(i/dim_x)
			i_x = i - i_y*dim_x
			self.plot(self.var_keys[i], ax=axs[i_y,i_x], do_show=False)

		fig_size_x = 4.5 * dim_x;
		fig_size_y = 2.7 * dim_y;
		fig.set_size_inches(fig_size_x,fig_size_y)
		plt.tight_layout()
		
		if do_show:
			plt.show()


	def filter_data_frame(self, var_key, min_val, max_val, keep_data = False):
		data_filter = self.data.loc[(self.data[var_key] >= min_val) & (self.data[var_key] <= max_val)] 	
		if keep_data:
			data = data_filter
		return data_filter

if __name__ == '__main__':
	
	filein_path_name = "./devel/test/elist/data/EventListExt.advelist"
	elist_1 = elist(filein_path_name)
	elist_1.print()
	# elist_1.plot("E")
	elist_1.plot_all()


	# filein_path_name = "./devel/test/elist/data/ExtElist.txt"
	# elist_2 = elist(filein_path_name)
	# elist_2.print()
	# # elist_2.plot("E")	
	# elist_2.plot_all()