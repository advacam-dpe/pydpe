import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from os import path
import re
import sys
import pandas as pd
from tqdm import tqdm
from matplotlib.colors import LogNorm
import copy

import hist1d as ht1d
import cluster as cl




class Clist(object):
    """docstring for clist"""
    def __init__(self, filein_path_name = "", usecols=None, nrows=None, do_print=True):
        super(Clist, self).__init__()

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
                    self.var_keys =  list(copy.deepcopy(self.data.keys())) 
                    self.ncol = len(self.var_keys)
                    self.nrow = len(self.data)
                else:
                    print("Load of data failed from file:" + file_path_name)

        except Exception as e:
            print(f"Can not open file: {file_path_name}. {e}")    
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
        print("clist")
        if len(self.filein_path_name): print("File:", self.filein_path_name)
        print("VarKeys:", self.var_keys)
        print("VarUnits:", self.var_units)
        print(self.data) 

    def plot(self, var_key, nbin = 100, do_show = True, ax=None,
            do_log_x = False, do_log_y = False, xmin=None, xmax=None, color="C0",
            label=""):

        if xmin is None or xmax is None: 
            max_val = np.max(self.data[var_key])
            min_val = np.min(self.data[var_key])
            range_val = max_val - min_val
            if min_val == max_val:
                range_val = abs(max_val)
                if min_val == 0:
                    range_val = 10
            if xmin is None:
                xmin = min_val - range_val*0.1
            if xmax is None:
                xmax = max_val + range_val*0.1

        hist1d_1 = ht1d.hist1d(nbin = nbin, xmin = xmin, xmax = xmax)

        hist1d_1.fill_np(self.data[var_key])
        hist1d_1.title = "Histogram of " + var_key
        hist1d_1.name = var_key if not label else label     
        hist1d_1.axis_labels = [var_key, "N"]
        hist1d_1.plot(do_show=do_show, ax=ax, do_log_y = do_log_y, do_log_x = do_log_x, color=color)

        ax.set_ylabel("N [-]", fontsize=10)

        unit_x = " "
        try:
            unit_x += f"[{self.var_units[list(self.var_keys).index(var_key)]}]"
        except:
            pass
        ax.set_xlabel(var_key + unit_x, fontsize=10)

        return hist1d_1

    def plot_all(self, nbin = 100, do_show = True):

        dim_x = int(math.sqrt(float(self.ncol)))
        dim_y = int(dim_x + 1)

        fig, axs = plt.subplots(dim_y, dim_x, sharex=False)

        for i in range(self.ncol):
            i_y = int(i/dim_x)
            i_x = i - i_y*dim_x
            if self.var_keys[i] != "ClusterID" and self.var_keys[i] != "ClusterPixels":
                try:
                    self.plot(self.var_keys[i], ax=axs[i_y,i_x], do_show=False)
                except Exception as e:
                    print(f"[ERROR] Fail to show histogram {self.var_keys[i]}: {e}.")

        fig_size_x = 4.5 * dim_x;
        fig_size_y = 2.7 * dim_y;
        fig.set_size_inches(fig_size_x,fig_size_y)
        plt.tight_layout()
        
        if do_show:
            plt.show()

    def plot_hist2d(self, var_key_x, var_key_y, fig=None, ax=None, do_show=True,
                    do_log_z = True):

        if not fig or not ax:
            fig, ax = plt.subplots()

        norm = None
        if do_log_z:
            norm = LogNorm()

        hist2d = ax.hist2d(self.data[var_key_x], self.data[var_key_y], bins=[100,100],    
                        norm = norm)

        cbar = plt.colorbar(hist2d[3], ax=ax)
        cbar.set_label("N [-]")

        unit_x = " "
        unit_y = " "

        try:
            unit_x += f"[{self.var_units[list(self.var_keys).index(var_key_x)]}]"
            unit_y += f"[{self.var_units[list(self.var_keys).index(var_key_y)]}]"
        except:
            pass

        ax.set_xlabel(var_key_x + unit_x)
        ax.set_ylabel(var_key_y + unit_y)

        if do_show:
            cbar = plt.colorbar(hist2d[3], ax=ax)            
            plt.show()

        return hist2d, cbar


    def plot_clusters(self, fig=None, ax=None, cluster_count=30, do_show=True, idx_start=None):
        if not fig or not ax:
            fig, ax = plt.subplots()

        i = 0
        idx = 0 
        hist = None
        if idx_start and idx_start > 0 and idx_start < len(self.data):
            idx = idx_start

        while i < cluster_count and idx < len(self.data):
            try:
                idx += 1
                cluster = self.get_cluster(cluster_idx=idx)
                hist, cbar = cluster.plot(fig=fig, ax=ax, show_plot=False)
                cbar.remove()
            except Exception as e:
                print(f"{e}")
                continue
            i += 1

        if hist is None:
            print("[ERROR] No clusters showed in plot_clusters fuction.")
            return

        ax.set_xlim(0,256)
        ax.set_ylim(0,256)
        cbar = plt.colorbar(hist[3], ax=ax)

        if do_show:
            cbar = plt.colorbar(hist[3], ax=ax)
            plt.show()

        return cbar

    def filter_data_frame(self, var_key, min_val, max_val, keep_data = False, get_out_data = False):
        data_filter = self.data.loc[(self.data[var_key] >= min_val) & (self.data[var_key] <= max_val)]  
        data_filter_out = None
        if get_out_data:
            data_filter_out = self.data.loc[(self.data[var_key] < min_val) | (self.data[var_key] > max_val)] 
        
        if keep_data:
            self.data = data_filter

        return data_filter, data_filter_out

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

        t_coinc = 100     #ns
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

    def get_cluster(self, cluster_id=None, cluster_idx=None):

        try:
            if cluster_id != None:            
                cluster_str = self.data.loc[self.data["ClusterID"]==cluster_id, "ClusterPixels"]
                cluster_str = cluster_str.iloc[0] # Cluster ID is not unique, take first instance
            elif cluster_idx != None:
                cluster_str = self.data.iloc[cluster_idx]["ClusterPixels"]

            cluster = cl.Cluster()

            cluster.load_from_string(cluster_str)
            return cluster
        except Exception as e:
            print(f"[ERROR] Failed to get cluster: at {cluster_id} or {cluster_idx}: {e}")
            return None

    def export(self, file_out_path_name):
        if self.data is None or self.data.empty:
            print("Can not export clist because it is empty.")
            return

        try:
            file_out = open(file_out_path_name, "w")

            if not file_out_path_name or file_out.closed:
                print("Cna not open file for export: ", file_out_path_name)
                return

            # header
            msg = ""
            for var_key in self.var_keys:
                msg += var_key + self.separator
            msg = msg[:-1] + '\n'
            file_out.write(msg)

            msg = ""
            for var_unit in self.var_units:
                msg += var_unit + self.separator
            msg = msg[:-1] + '\n'
            file_out.write(msg)

            # data
            for i,row in self.data.iterrows():
                msg = ""
                for num in row:
                    msg += str(num) + self.separator
                msg = msg[:-1] + '\n'
                file_out.write(msg)

            file_out.close()
        except Exception as e:
            print(f"Fail to export clist: {e}.")

    def extend_varaibles(self):
        if self.data is None or self.data.empty:
            print("[ERROR] Failed ot extend clist because there is no data.")
            return

        try:
            # epix relative std
            self.data["EpixStdRel"] = self.data["EpixStd"] / self.data["EpixMean"]
            self.var_keys.append("EpixStdRel")
            self.var_units.append("-")

            self.data["EpixMean/Height"] = self.data["EpixMean"] / self.data["Height"]
            self.var_keys.append("EpixMean/Height")
            self.var_units.append("-")

            self._calculate_cluster_pixel_variable_extensions()

            # self.data["EpixMean/Height"] = self.data["EpixMean"] / self.data["Height"]
            # self.var_keys.append("EpixMean/Height")
            # self.var_units.append("-")

        except Exception as e:
            print(f"[ERROR] Failed to extend clist: {e}.")

    def _calculate_cluster_pixel_variable_extensions(self):
        if self.data is None or self.data.empty:
            print("[ERROR] Failed to calculate new cluster variables because there is no data.")
            return

        for idx in  range(len(self.data)):
            cluster = self.get_cluster(cluster_idx=idx)     
            self._calculate_cluster_energy_levels(cluster)   


    def _calculate_cluster_energy_levels(self, cluster):
        cluster.convert_pixels_to_numpy()

        energy_intervals = [30, 330, 1e100]
        
        energy_sums = []
        count_sums = []

        energy = 0
        size = 0
        for i in range(len(energy_intervals)):
            energy_sums.append(0)
            count_sums.append(0)

        for row in cluster.pixels_np:
            px_energy = row[2]

            energy += px_energy
            size += 1

            for idx, energy_interval in enumerate(energy_intervals):
                if px_energy < energy_interval:
                    energy_sums[idx] += px_energy
                    count_sums[idx] += 1
                    break

        energy_sums_fraction = [num / energy for num in energy_sums]
        count_sums_fraction = [num / size for num in count_sums]

        print("-----------------------")
        print("energy_sums", energy_sums)
        print("energy_sums_fraction", energy_sums_fraction)
        print("count_sums", count_sums)
        print("count_sums_fraction", count_sums_fraction)



if __name__ == '__main__':
    
    case = 6

    if case == 1:

        filein_path_name = "./devel/test/clist/data/elist_t3pa.clist"
        clist_1 = Clist(filein_path_name)
        clist_1.print()
        # clist_1.plot("E")
        clist_1.plot_all()

    elif case == 2:

        filein_path_name = "./devel/test/clist/data/Extclist.txt"
        clist_2 = Clist(filein_path_name)
        clist_2.print()
        # clist_2.plot("E")    
        clist_2.plot_all()

    elif case == 3:

        filein_path_name = "/mnt/MainDisk/Soubory/Programy/Vlastni/c++/aplikace/DataProcessing/PreProcessing/Coincidence_Matching/Devel/Test/LargeFile/clistAll.advclist"
        clist_1 = Clist(filein_path_name)
        clist_1.print()
        is_increasing = clist_1.data["T"].is_monotonic_increasing
        print(is_increasing)
        coinc_cml = clist_1.analyse_coincidence()
        print(len(coinc_cml), len(clist_1.data["T"]))

    elif case == 4:

        # filein_path_name = "./devel/test/clist/data/CListExt.clist"
        filein_path_name = "D:\\db\\Data\\Single\\HeavyIons_Hadrons\\01p\\F_III_XMeV_AdvapixTPX3_H09_500Si_p80V_BS\\08MeV\\33_30deg\\File\\CListExt.clist"

        clist = Clist(filein_path_name)

        cluster_377 =  clist.get_cluster(377)

        # print(cluster_377)

        cluster_377.plot()

    elif case == 5:

        filein_path_name = "./devel/in/CListExt.clist"
        clist_2 = Clist(filein_path_name, nrows=1000)
        clist_2.print()
        # clist_2.plot("E")    
        clist_2.plot_clusters()

    elif case == 6:

        filein_path_name = "./devel/in/CListExt.clist"
        file_out_path_name = "./devel/data.clist"
        clist_2 = Clist(filein_path_name, nrows=1000)
        # clist_2.print()
        # clist_2.plot("E")    
        # clist_2.plot_clusters()

        clist_2.export(file_out_path_name)






