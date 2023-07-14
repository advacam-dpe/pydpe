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
from matplotlib.colors import LogNorm



class t3pa(object):
	"""docstring for t3pa"""
	def __init__(self, filein_path_name = ""):
		super(t3pa, self).__init__()

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
					with tqdm(total=total_rows-2, unit='line', unit_scale=True, desc='Loading data') as pbar:	
						for chunk in pd.read_csv(file_path_name, chunksize=chunk_size, header=[0], sep=self.separator):
							chunks.append(chunk)
							pbar.update(len(chunk))

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

		index_max = np.max(self.data["Matrix Index"]) + 1
		dim = 256

		if index_max > 0:
			dim = int(math.sqrt(index_max))

		self.data['x'] = self.data['Matrix Index']%dim
		self.data['y'] = self.data['Matrix Index']//dim	

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

	def print(self):
		print("t3pa")
		if len(self.filein_path_name): print("File:", self.filein_path_name)
		print("VarKeys:", self.var_keys)
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
			self.data = data_filter
		return data_filter

	def sensor_map(self, val_index ):

		if val_index > self.nrow-1:
			return np.array([[-1]])

		dim = 256

		index_max = np.max(self.data["Matrix Index"]) + 1

		if index_max > 0:
			dim = int(math.sqrt(index_max))

		matrix = np.zeros((dim, dim))

		for index, row in self.data.iterrows():
			matrix[row[7],row[6]] += row[val_index]

		return matrix


	def plot_matrix(self, matrix, file_out_path_name, do_log_z = False, names = [], values = []):
		
		# fig, ax = plt.subplots()
		fig = plt.figure(num=None, facecolor='w', edgecolor='k')
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

def PlotGraph1D(ListX, ListY):
	#Main plot function
	plt.plot(ListX, ListY, color='gainsboro', linewidth=0,
		marker='o', markerfacecolor='dodgerblue', markeredgewidth=0,
		markersize=1)
	plt.show()


if __name__ == '__main__':
	
	# filein_path_name = "./devel/test/t3pa/in/MASK_tot_toa.t3pa"
	# filein_path_name = "/mnt/MainDisk/Soubory/Programy/Vlastni/c++/aplikace/DataProcessing/Processing/DPE/Test/out/test_009/File/MASK_tot_toa.t3pa"
	# t3pa_file = t3pa(filein_path_name)
	# t3pa_file.print()
	# # t3pa_file.plot("Matrix Index")
	# # t3pa_file.plot_all()
	# sensor_matrix = t3pa_file.sensor_map(3)
	# t3pa_file.plot_matrix(sensor_matrix, "", True)


	t3pa_file = t3pa("/mnt/MainDisk/Soubory/Analysis/MinipixToA/data/l06_co60_60s_thl5keV.t3pa")
	t3pa_file.print()


	PlotGraph1D(t3pa_file.data["Index"], t3pa_file.data["ToA"]*25)

