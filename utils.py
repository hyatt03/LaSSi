# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def cross(a, b):
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]

    return np.array([x, y, z])

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def downsample(a, r):
    nlen = len(a) / r  # number of chunks
    npad = len(a) % nlen  # apply some padding if the data is not dividable

    if npad:
        a = np.append(np.zeros((nlen - npad)) * np.NaN, a)  # append the neccessary padding values as nan

    b = a.reshape(-1, nlen)  # create the chunks

    # calculate mean and max for each chunk and zip it together with reshape in fortran mode
    chunked = np.vstack((np.nanmax(b, axis=1), np.nanmin(b, axis=1))).reshape(-1, 1, order='F')

    return pd.DataFrame(chunked)[0]
