# -*- coding: utf-8 -*-

import scipy.fftpack as sci
from scipy import linspace

def fourier(options, timeseries):
    Fs = options.dt
    L = len(timeseries)

    Y = sci.fftn([timeseries.pos_x, timeseries.pos_y, timeseries.pos_z])

    x_f, y_f, z_f = Y[0], Y[1], Y[2]
    total_fourier = [(x + y + z) / 9 for x, y, z, in zip(x_f, y_f, z_f)]

    the_list_1 = linspace(0, int((L / 2 - 1)), int(L / 2))
    the_list_2 = [x / L for x in the_list_1]
    f = [x * Fs for x in the_list_2]

    energy = [x * options.GHz_to_meV / (10 ** 9) for x in f[0:len(total_fourier)]]

    return total_fourier, f, energy

def plot_fourier(total_fourier, f, energy):

    pass
