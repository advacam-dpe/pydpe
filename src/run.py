import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from os import path
import re
import sys
import time
import multiprocessing
import subprocess


dpe = "/mnt/MainDisk/Soubory/Programy/Vlastni/c++/aplikace/DataProcessing/Processing/DPE/Release/Linux/dpe.sh"

param_file_paths = ["/run/media/lukasm/10ABE5A17D6AED39/Work/Data/Single/MixedFields/RadiationSources/133Ba_Source/B_MiniTPX3_H09_2mmCdTe_n450V/",
					"/run/media/lukasm/10ABE5A17D6AED39/Work/Data/Single/MixedFields/RadiationSources/137Cs_Source/C_MiniTPX3_H09_2mmCdTe_n450V/",
					"/run/media/lukasm/10ABE5A17D6AED39/Work/Data/Single/MixedFields/RadiationSources/152Eu_Source/B_MiniTPX3_H09_2mmCdTe_n450V/",
					"/run/media/lukasm/10ABE5A17D6AED39/Work/Data/Single/MixedFields/RadiationSources/22Na_Source/B_MiniTPX3_H09_2mmCdTe_n450V/",
					"/run/media/lukasm/10ABE5A17D6AED39/Work/Data/Single/MixedFields/RadiationSources/241Am_Source/I_MiniTPX3_H09_2mmCdTe_n450V/",
					"/run/media/lukasm/10ABE5A17D6AED39/Work/Data/Single/MixedFields/RadiationSources/60Co_Source/D_MiniTPX3_H09_2mmCdTe_n450V/"]

param_file_name = "ParametersFile.txt"

do_multi_process = True

def cmd_dpe(dpe, param_file_path, param_file_name):

	cmd = dpe + " " + param_file_path + param_file_name
	os.system(cmd)

def run_dpe():
	if do_multi_process:

		processes = []
		for n in range(len(param_file_paths)):
			print("Process num: ", n )
			process = multiprocessing.Process(target=cmd_dpe, args=(dpe,param_file_paths[n],param_file_name,))
			processes.append(process)
			process.start()

		for process in processes:
			process.join()
	else:

		cmd = dpe + " " + param_file_paths[0] + param_file_name
	os.system(cmd)


if __name__ == '__main__':
    run_dpe()
	