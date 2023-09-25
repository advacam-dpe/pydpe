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
from tqdm import tqdm

class clog(object):
	"""docstring for clog"""
	def __init__(self, filein_path_name = "", usecols=None, nrows=None, do_print=True):
		super(clog, self).__init__()

		rc = 0

		self.basic_init()

		if len(filein_path_name) != 0:
			rc = self.load(filein_path_name, usecols, nrows, do_print)

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


	def load(self, file_path_name, usecols=None, nrows=None, do_print=True):
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

					# get the total number of rows in the CSV file
					total_rows = pd.read_csv(file_path_name, nrows=0).shape[0]

					# define the chunk size for reading the CSV file
					chunk_size = 10000

					# initialize an empty list to store the chunks
					chunks = []

					# iterate over the CSV file by reading it in chunks
					if do_print:
						with tqdm(total=total_rows-2, unit='line', unit_scale=True, desc='Loading data') as pbar:	
							for chunk in pd.read_csv(file_path_name, chunksize=chunk_size, header=0, 
														skiprows = [1], sep=self.separator, usecols=usecols,
														nrows=nrows, skip_blank_lines=True):
								chunk.dropna(axis=0, inplace=True)								
								chunks.append(chunk)
								pbar.update(len(chunk))
					else:
						for chunk in pd.read_csv(file_path_name, chunksize=chunk_size, header=0, 
													skiprows = [1], sep=self.separator, usecols=usecols,
													nrows=nrows, skip_blank_lines=True):
							chunk.dropna(axis=0, inplace=True)
							chunks.append(chunk)						
					
					# concatenate the chunks into a single dataframe
					self.data = pd.concat(chunks, ignore_index=True)

					# self.data = pd.read_csv(file_path_name, sep=self.separator, header=[0], skiprows = [1])
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

		# print("|" + separator + "|")

		if len(separator) != 0:
			self.separator = separator
		return separator

	def __load_units(self, line):
		line = line.replace("\n", "")
		self.var_units = line.split(self.separator)

	def print(self):
		print("clog")
		if len(self.filein_path_name): print("File:", self.filein_path_name)
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
			if self.var_keys[i] != "ClusterID" and self.var_keys[i] != "ClusterPixels":
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
			self.data = data_filter
		return data_filter

	def randomize(self, do_reset_idx = False, keep_data = False):
		data_rand = pd.DataFrame()

		if not do_reset_idx:
			data_rand = self.data.sample(frac=1)
		else:
			data_rand = self.data.sample(frac=1).reset_index(drop=True)

		if keep_data:
			data = data_rand
		return data_rand

	def analyse_coincidence(self):

		coinc_clm = np.empty(self.nrow)
		coinc_clm[0] = 0

		t_coinc = 100 	#ns
		t_curr = -1
		t_main = self.data.loc[0, "T"] + t_coinc + 1
		coinc_id_curr = 0

		correct = True
		do_comp = False
		if "EventID" in self.var_keys:
			do_comp = True

		for index, row in self.data.iterrows():
			if index == 0:
				continue

			t_curr = row[5]
			if t_curr - t_main > t_coinc:
				coinc_id_curr += 1
				t_main = t_curr
			
			# print(index, t_curr, t_main, coinc_id_curr, int(row[1]), coinc_id_curr - int(row[1]))

			if do_comp and coinc_id_curr - int(row[1]) != 0:
				break
				correct = False

			coinc_clm[index] = coinc_id_curr

		if do_comp and correct:
			print("Event ID is already correct with coinc window ", t_coinc)

		return coinc_clm


if __name__ == '__main__':
	
	filein_path_name = "./devel/test/clog/data/elist_t3pa.clog"
	clog_1 = clog(filein_path_name)
	clog_1.print()
	# clog_1.plot("E")
	clog_1.plot_all()


	# filein_path_name = "./devel/test/clog/data/Extclog.txt"
	# clog_2 = clog(filein_path_name)
	# clog_2.print()
	# # clog_2.plot("E")	
	# clog_2.plot_all()


	# filein_path_name = "/mnt/MainDisk/Soubory/Programy/Vlastni/c++/aplikace/DataProcessing/PreProcessing/Coincidence_Matching/Devel/Test/LargeFile/clogAll.advclog"
	# clog_1 = clog(filein_path_name)
	# clog_1.print()
	# is_increasing = clog_1.data["T"].is_monotonic_increasing
	# print(is_increasing)
	# coinc_cml = clog_1.analyse_coincidence()
	# print(len(coinc_cml), len(clog_1.data["T"]))








