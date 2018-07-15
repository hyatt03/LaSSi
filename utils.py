# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math


def cross(a, b):
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]

    return np.array([x, y, z])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def to_sph(a):
    r = math.sqrt(dot(a, a))
    theta = math.atan2(math.sqrt(a[0] ** 2 + a[1] ** 2), a[2])
    phi = math.atan2(a[1], a[2])

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
