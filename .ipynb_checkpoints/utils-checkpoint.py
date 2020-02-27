# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
h = 4.13566733e-15


def cross(a, b):
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]

    return np.array([x, y, z])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def to_sph(a):
    r = np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    theta = math.acos(a[2] / r)
    phi = math.atan2(a[1], a[0])

    return r, theta, phi


def to_cart(a):
    x = a[0] * math.sin(a[1]) * math.cos(a[2])
    y = a[0] * math.sin(a[1]) * math.sin(a[2])
    z = a[0] * math.cos(a[1])

    return np.array([x, y, z])


def downsample(a, r):
    nlen = int(len(a) / r)  # number of chunks
    npad = len(a) % nlen  # apply some padding if the data is not dividable

    if npad:
        a = np.append(np.zeros((nlen - npad)) * np.NaN, a)  # append the neccessary padding values as nan

    b = a.reshape(-1, nlen)  # create the chunks

    # calculate mean and max for each chunk and zip it together with reshape in fortran mode
    chunked = np.vstack((np.nanmax(b, axis=1), np.nanmin(b, axis=1))).reshape(-1, 1, order='F')

    return pd.DataFrame(chunked)[0]


# Takes E in meV
# Returns dt and N
def calculate_dt_and_n(energy_min, energy_max):
    # Convert energy to frequency using Plancks constant
    f_min = energy_min / (h * 1e3)
    f_max = energy_max / (h * 1e3)

    # We need at least 1000 datapoints per revolution for the highest energy
    dt = 1 / (f_max * 1e3)

    # And we need at least 20 periods for the low energy
    t_tot = 20 / f_min
    n = t_tot / dt

    return dt, n
