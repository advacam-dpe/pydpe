import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from math import factorial
import sys
sys.path.append("/mnt/MainDisk/Soubory/Programy/Vlastni/python/aplikace/advacam/dpe/src/")
import fit as ft

class hist1d(object):
     """docstring for hist1d"""
     def __init__(self, low_edges = np.array([0.0]), xmin=-1, xmax=-1, nbin=0 , title = "", name = "",
                    axis_labels = []):
          super(hist1d, self).__init__()

          self.basic_init()

          if len(low_edges) > 1:
               self.low_edges = low_edges # low edges from xmin also including the xmax
               self.bin_conts = np.zeros(shape = [len(self.low_edges)-1], dtype = float)
               self.nbin = len(self.low_edges) - 1
               self.xmin = self.low_edges[0]
               self.xmax = self.low_edges[len(low_edges)-1]
               self.bin_width = self.low_edges[1] - self.low_edges[0]
               self.is_eq_width =  self.check_eq_bin_width()
               self.done_init = True

          elif xmin < xmax and nbin > 0:
               self.xmin = xmin
               self.xmax = xmax
               self.nbin = nbin
               self.bin_width = (self.xmax - self.xmin)/float(self.nbin)
               hist_low_edges = np.arange(self.xmin, self.xmax + self.bin_width, self.bin_width)       
               self.low_edges = hist_low_edges # low edges from xmin also including the xmax
               self.bin_conts = np.zeros(shape = [len(self.low_edges)-1], dtype = float)
               self.is_eq_width =  True
               self.done_init = True

          if len(title) != 0:
               self.title = title
          if len(name) != 0:
               self.name = name
          if len(axis_labels) == 2:
               self.axis_labels = axis_labels

          self.x_range = [xmin, xmax]
          self.x_range_width = xmax - xmin
          self.ymax = 0

     def basic_init(self):
          self.done_init = False

          self.low_edges = np.empty(shape = [1], dtype = float) # low edges from xmin also including the xmax
          self.bin_conts = np.zeros(shape = [1], dtype = float)
          self.nbin = 0
          self.xmin = -1
          self.xmax = -1
          
          self.bin_width = 0
          self.is_eq_width =  False

          self.title = "hist_title"
          self.name = "hist_name"
          self.axis_labels = ["X", "Y"] 

          self.integ = 0

     #---------------------------------
     #Functions 
     #---------------------------------

     def check_eq_bin_width(self):
          if self.bin_width > 0:
               for i in range(len(self.low_edges)-1):
                    if self.low_edges[i+1] - self.low_edges[i] != self.bin_width:
                         self.is_eq_width = False
                         self.bin_width = -1
                         return False
          self.is_eq_width = True                         
          return True

     def fill(self, num):
          return

     def fill_np(self, array):
          hist_bin_conts, hist_low_edges = np.histogram(array, bins = self.low_edges)
          if len(self.bin_conts) == len(hist_bin_conts):
               self.bin_conts = self.bin_conts + hist_bin_conts

     def integral(self):
          integ = np.sum(self.bin_conts) 
          return integ

     def norm(self):     
          sum_bins = np.sum(self.bin_conts)
          if sum_bins != 0:
               self.bin_conts = self.bin_conts/sum_bins

     def norm_max(self): 
          max_bins = np.max(self.bin_conts)
          if sum_bins != 0:
               self.bin_conts = self.bin_conts/max_bins

     def shift(self, shift):
          if shift != 0 and is_eq_width:
               bin_width = self.low_edges[1] - self.low_edges[0]
               i_shift = int(shift/bin_width)

               for i in range(len(self.bin_conts)):
                    if i+i_shift < len(self.bin_conts):
                         self.bin_conts[i] = self.bin_conts[i + i_shift]
                    else:
                         self.bin_conts[i] = 0


     def bin_cont_max(self):
          self.ymax = np.max(self.bin_conts)
          return self.ymax

     def xrange(self):
          low_edges_nonzero = self.low_edges[np.append(self.bin_conts,0) != 0]
          self.x_range[0] = np.min(low_edges_nonzero)
          self.x_range[1] = np.max(low_edges_nonzero)
          self.x_range_width =  self.x_range[1] - self.x_range[0]
          return self.x_range

     def plot(self, color = "C0", do_show = True, do_zoom = False, do_log_x = False, do_log_y  = False):
          bin_conts_ext = np.append(self.bin_conts, 0)   

          plt.hist(x=self.low_edges, bins=self.low_edges, weights=bin_conts_ext, color=color, histtype='step', 
                         linewidth=1.6, label=self.name, alpha = 0.8)

          self.xrange()
          self.bin_cont_max()

          plt.ylim(ymin=0, ymax = self.ymax*1.2) 
          plt.xlim(xmin=self.x_range[0]-0.1*self.x_range_width, xmax = self.x_range[1]+0.1*self.x_range_width)
          plt.legend()
          plt.grid(visible = True, color ='grey',  linestyle ='-.', linewidth = 0.5,  alpha = 0.6)
          plt.xlabel(self.axis_labels[0], fontsize=12) 
          plt.ylabel(self.axis_labels[1], fontsize=12) 
          plt.title(self.title)    
          if do_log_y:
               plt.yscale('log')
          if do_log_x:
               plt.xscale('log') 

          if do_show:
               plt.show()          

     def smooth_filter(self, nums, n = 10):
          nums_smooth = np.empty(shape=[len(nums)])

          for i in range(0,len(nums)):
               bin_cont_sum = 0
               for j in range(-n,n+1):
                    if i+j > 0 and i+j < len(nums):
                         bin_cont_sum += nums[i+j]
               nums_smooth[i] = float(bin_cont_sum)/float(n)

          nums_smooth = nums_smooth*(np.sum(nums)/np.sum(nums_smooth))

          return nums_smooth

     def smooth_mean(self, n = 10, keep_new_bins = True):
          hist_bins_smooth = np.empty(shape=[len(self.bin_conts)])
          
          for i in range(len(self.bin_conts)):
               hist_bins_smooth[i] = self.bin_conts[i]          

          for i in range(n):
               hist_bins_smooth = self.smooth_filter(hist_bins_smooth, 1)

          if keep_new_bins:
               self.bin_conts = hist_bins_smooth

          return hist_bins_smooth

     def savitzky_golay(self, y, window_size, order, deriv=0, rate=1):
         r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
         The Savitzky-Golay filter removes high frequency noise from data.
         It has the advantage of preserving the original shape and
         features of the signal better than other types of filtering
         approaches, such as moving averages techniques.
         Parameters
         ----------
         y : array_like, shape (N,)
             the values of the time history of the signal.
         window_size : int
             the length of the window. Must be an odd integer number.
         order : int
             the order of the polynomial used in the filtering.
             Must be less then `window_size` - 1.
         deriv: int
             the order of the derivative to compute (default = 0 means only smoothing)
         Returns
         -------
         ys : ndarray, shape (N)
             the smoothed signal (or it's n-th derivative).
         Notes
         -----
         The Savitzky-Golay is a type of low-pass filter, particularly
         suited for smoothing noisy data. The main idea behind this
         approach is to make for each point a least-square fit with a
         polynomial of high order over a odd-sized window centered at
         the point.
         Examples
         --------
         t = np.linspace(-4, 4, 500)
         y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
         ysg = savitzky_golay(y, window_size=31, order=4)
         import matplotlib.pyplot as plt
         plt.plot(t, y, label='Noisy signal')
         plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
         plt.plot(t, ysg, 'r', label='Filtered signal')
         plt.legend()
         plt.show()
         References
         ----------
         .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
            Data by Simplified Least Squares Procedures. Analytical
            Chemistry, 1964, 36 (8), pp 1627-1639.
         .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
            W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
            Cambridge University Press ISBN-13: 9780521880688
         """    
         try:
             window_size = np.abs(int(window_size))
             order = np.abs(int(order))
         except ValueError:
             raise ValueError("window_size and order have to be of type int")
         if window_size % 2 != 1 or window_size < 1:
             raise TypeError("window_size size must be a positive odd number")
         if window_size < order + 2:
             raise TypeError("window_size is too small for the polynomials order")
         order_range = range(order+1)
         half_window = (window_size -1) // 2
         # precompute coefficients
         b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
         m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
         # pad the signal at the extremes with
         # values taken from the signal itself
         firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
         lastvals = y[-1] + np.abs( y[-half_window-1:-1][::-1] - y[-1] )
         y = np.concatenate((firstvals, y, lastvals))
         return np.convolve( m[::-1], y, mode='valid')

     def smooth_pol(self, window=11, pol_order=3, keep_new_bins = True):
          bin_conts_smooth = self.savitzky_golay(self.bin_conts, window, pol_order) # window size 51, polynomial order 3
          if keep_new_bins:
               self.bin_conts = bin_conts_smooth
          return bin_conts_smooth

     def fit_gauss(self, A = None, mu = None, sigma = None, x_range = [-1,-1]):
          x = self.low_edges
          y = np.append(self.bin_conts,0)     
          *popt, pcov, perr, x_fit, y_fit = ft.fit_gauss(x, y, A, mu, sigma, x_range = x_range)
          return *popt, pcov, perr, x_fit, y_fit

     def fit_gerf(self, A1 = None, A2 = None, mu = None, sigma = None, x_range = [-1,-1]):
          x = self.low_edges
          y = np.append(self.bin_conts,0)     
          *popt, pcov, perr, x_fit, y_fit = ft.fit_gerf(x, y, A1, A2, mu, sigma, x_range = x_range)
          return *popt, pcov, perr, x_fit, y_fit

     def fit_gerfc(self, A1 = None, A2 = None, mu = None, sigma = None, x_range = [-1,-1]):
          x = self.low_edges
          y = np.append(self.bin_conts,0)     
          *popt, pcov, perr, x_fit, y_fit = ft.fit_gerfc(x, y, A1, A2, mu, sigma, x_range = x_range)
          return *popt, pcov, perr, x_fit, y_fit

#---------------------------------
#Main processing part
#---------------------------------

if __name__ == '__main__':

     case = 20

     if case == 0:

          n_points = 10000
          mu, sigma = 50, 10 
          data = np.random.normal(mu, sigma, n_points)
          mu, sigma = 20, 5 
          data = np.append(data, np.random.normal(mu, sigma, n_points))

          hist1 = hist1d(xmin = 0, xmax = 100, nbin = 100)
          hist1.name = "raw" 
          hist1.title = "Smootinhg of hist with polynom and mean methods"      
          hist1.fill_np(data)
          hist1.plot(color = "C0")

     if case == 20:
          n_points = 100000
          mu, sigma = 50, 10 
          data = np.random.normal(mu, sigma, n_points)
          mu, sigma = 20, 5 
          data = np.append(data, np.random.normal(mu, sigma, n_points))

          hist1 = hist1d(xmin = 0, xmax = 100, nbin = 100)
          hist1.fill_np(data)
          hist1.plot(color = "C1", do_show = False)

          *popt, pcov, perr, x_fit, y_fit = hist1.fit_gauss(x_range = [35, 80])
          print(popt)
          print(perr)
          print(perr*100./popt)
          plt.plot(x_fit, y_fit, '-', label='fit 1')

          *popt, pcov, perr, x_fit, y_fit = hist1.fit_gauss(x_range = [0, 30])
          print(popt)        
          print(perr)           
          print(perr*100./popt)
          plt.plot(x_fit, y_fit, '-', label='fit 2', color = "C3")

          plt.legend()
          plt.show()          



     if case == 100:
          smooth_mean_n = 20

          smooth_pol_window = 21
          smooth_pol_polord = 3

          min_val = 0.
          max_val = 100.
          nbin = 100
          bin_width = (max_val - min_val)/float(nbin)
          hist_low_edges = np.arange(min_val, max_val + bin_width, bin_width)          
          n_points = 100

          mu, sigma = 50, 10 
          data = np.random.normal(mu, sigma, n_points)
          mu, sigma = 20, 5 
          data = np.append(data, np.random.normal(mu, sigma, n_points))

          hist2 = hist1d(hist_low_edges) 
          hist2.name = "smooth mean: " + str(smooth_mean_n)
          hist2.fill_np(data)
          hist2.smooth_mean(smooth_mean_n)
          hist2.plot(color = "C1", do_show = False)

          hist3 = hist1d(hist_low_edges) 
          hist3.name = "smooth pol: r = " + str(smooth_pol_polord) + " w = " + str(smooth_pol_window)
          hist3.fill_np(data)
          hist3.smooth_pol(smooth_pol_window, smooth_pol_polord)
          hist3.plot(color = "C3", do_show = False)

          hist1 = hist1d(xmin = min_val, xmax = max_val, nbin = nbin)
          hist1.name = "raw" 
          hist1.title = "Smootinhg of hist with polynom and mean methods"      
          hist1.fill_np(data)
          hist1.plot(color = "C0")