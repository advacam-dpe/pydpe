import numpy as np
import math
import matplotlib.pyplot as plt


def load_calib_matrixes(cal_mat_dir, cal_mat_a, cal_mat_b, cal_mat_c, cal_mat_t):

    cols = list(range(0,256))
    cal_mat_a = np.loadtxt(cal_mat_dir + "a.txt", delimiter=" ", usecols =cols)
    cal_mat_b = np.loadtxt(cal_mat_dir + "b.txt", delimiter=" ", usecols =cols)
    cal_mat_c = np.loadtxt(cal_mat_dir + "c.txt", delimiter=" ", usecols =cols)
    cal_mat_t = np.loadtxt(cal_mat_dir + "t.txt", delimiter=" ", usecols =cols)

def calibrate_pixel(pixel, cal_mat_a, cal_mat_b, cal_mat_c, cal_mat_t):
    x = pixel[0]
    y = pixel[1]
    tot = pixel[2]
    energy = 0

    A = cal_mat_a[y,x]
    T = cal_mat_t[y,x]
    B = cal_mat_b[y,x] - A*cal_mat_t[y,x] - tot
    C = T*tot - cal_mat_b[y,x]*T - cal_mat_c[y,x]
    if A != 0 and (B*B-4.0*A*C) >= 0:
        energy = (-B + math.sqrt(B*B - 4.0*A*C))/2.0/A
    else:
        energy = 0
    if energy < 0:
        energy = 0

    pixel[2] = energy

    return energy



if __name__ == '__main__':

    cal_mat_dir = "/home/lukas/file/sw/cpp/data_proc/proc/dpe/test/data/test_020/CalMat/"

    load_calib_matrixes(cal_mat_dir, cal_mat_a, cal_mat_b, cal_mat_c, cal_mat_t)

    plt.imshow(cal_mat_a)
    plt.show()

    calibrate_pixel([65,185,112], cal_mat_a, cal_mat_b, cal_mat_c, cal_mat_t)
    calibrate_pixel([62,184,131], cal_mat_a, cal_mat_b, cal_mat_c, cal_mat_t)
    calibrate_pixel([61,184,26], cal_mat_a, cal_mat_b, cal_mat_c, cal_mat_t)
