
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
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy import special

import hist1d as hs


def gauss(x,A,mu,sigma):
	return A*np.exp(-(x-mu)**2/(2*sigma**2))

def gerf(x,A1,A2,mu,sigma): 
	return A1*np.exp(-(x-mu)**2/(2*sigma**2)) + A2*special.erf((x-mu)/(np.sqrt(2)*sigma))

# erfc(x) = 1 - erf(x).
def gerfc(x,A1,A2,mu,sigma):
	return A1*np.exp(-(x-mu)**2/(2*sigma**2)) + A2*special.erfc((x-mu)/(np.sqrt(2)*sigma))
	
# erfc(x) = 1 - erf(x).
def erfc(x,A,mu,sigma): 
	return A*special.erfc((x-mu)/(np.sqrt(2)*sigma))

def estimate_gauss(x, y):
	mu = sum(x*y)/sum(y)
	sigma = np.sqrt(sum(y*(x-mu)**2)/sum(y))
	A = np.max(y)
	return A, mu, sigma

def fit_gauss(x, y, A = None, mu = None, sigma = None, x_range = [-1,-1]):

	x2 = x
	y2 = y

	if len(x_range) == 2 and x_range[0] < x_range[1]:
		x2 = x[(x < x_range[1]) & (x > x_range[0])]
		y2 = y[(x < x_range[1]) & (x > x_range[0])]

	A_est, mu_est, sigma_est = estimate_gauss(x2, y2)

	if A == None: 		A = A_est
	if mu == None: 		mu = mu_est 
	if sigma == None: 	sigma = sigma_est 

	# fit data
	x_fit = np.arange(min(x2), max(x2), 0.1)
	popt, pcov = curve_fit(gauss, x2, y2, p0=[A,mu,sigma]) #gauss
	y_fit = gauss(x_fit,*popt)
	perr = np.sqrt(np.diag(pcov))

	return *popt, pcov, perr, x_fit, y_fit


def fit_gerf(x, y, A1 = None, A2 = None, mu = None, sigma = None, x_range = [-1,-1]):

	x2 = x
	y2 = y

	if len(x_range) == 2 and x_range[0] < x_range[1]:
		x2 = x[(x < x_range[1]) & (x > x_range[0])]
		y2 = y[(x < x_range[1]) & (x > x_range[0])]

	A1_est, mu_est, sigma_est = estimate_gauss(x2, y2)

	if A1 == None: 		A1 = A1_est
	if mu == None: 		mu = mu_est 
	if sigma == None: 	sigma = sigma_est 
	if A2 == None:		A2 = 0;

	# fit data
	x_fit = np.arange(min(x2), max(x2), 0.1)
	popt, pcov = curve_fit(gerf, x, y, p0=[A1, A2, mu,sigma]) #gerfc
	y_fit = gerf(x_fit,*popt)
	perr = np.sqrt(np.diag(pcov))

	return *popt, pcov, perr, x_fit, y_fit

def fit_gerfc(x, y, A1 = None, A2 = None, mu = None, sigma = None, x_range = [-1,-1]):

	x2 = x
	y2 = y

	if len(x_range) == 2 and x_range[0] < x_range[1]:
		x2 = x[(x < x_range[1]) & (x > x_range[0])]
		y2 = y[(x < x_range[1]) & (x > x_range[0])]

	A1_est, mu_est, sigma_est = estimate_gauss(x2, y2)

	if A1 == None: 		A1 = A1_est
	if mu == None: 		mu = mu_est 
	if sigma == None: 	sigma = sigma_est 
	if A2 == None:		A2 = 0;

	# fit data
	x_fit = np.arange(min(x2), max(x2), 0.1)
	popt, pcov = curve_fit(gerfc, x, y, p0=[A1, A2, mu,sigma]) #gerfc
	y_fit = gerfc(x_fit,*popt)
	perr = np.sqrt(np.diag(pcov))

	return *popt, pcov, perr, x_fit, y_fit


if __name__ == '__main__':

	n_points = 10000
	mu, sigma = 50, 10 
	data = np.random.normal(mu, sigma, n_points)
	mu, sigma = 20, 5 
	data = np.append(data, np.random.normal(mu, sigma, n_points))

	hist1 = hs.hist1d(xmin = 0, xmax = 100, nbin = 100)
	hist1.fill_np(data)
	hist1.plot(color = "C1")


	xmin = 30
	xmax = 100
	A = 200
	mu = 50
	sigma = 10
	shift = 0

	x = hist1.low_edges
	y = np.append(hist1.bin_conts,0)

	*popt, pcov, perr, x_fit, y_fit = fit_gauss(x, y, x_range = [xmin, xmax])

	plt.plot(x_fit, y_fit, '-', label='fit')

	plt.show()
