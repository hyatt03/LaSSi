# -*- coding: utf-8 -*-

import numpy as np

def cross(a, b):
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]

    return np.array([x, y, z])

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
