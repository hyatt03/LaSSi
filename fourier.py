# -*- coding: utf-8 -*-

import numpy as np
from utils import dot
import cmath

def transform_on_q(q, options, constants, timeseries, particles, fourier_length):
    o, c = options, constants
    if o['debug']:
        print('Computing intensities for q = [{}, {}, {}]'.format(q[0], q[1], q[2]))

    # Sets the length to the next power of 2, for speed
    I_aa = np.zeros((3, int(fourier_length / 2)), dtype=np.dtype('complex128'))
    t_0 = 1

    for z in range(0, 3):
        for tablename, table in timeseries.items():
            particle = particles.get_atom_from_tablename(tablename)
            if type(table) is np.ndarray:
                positions = table
            else:
                positions = np.array([
                    table.cols.pos_x,
                    table.cols.pos_y,
                    table.cols.pos_z
                ], dtype=np.complex128)

            q_dot_lattice = cmath.exp(1j * dot(q, particle.lattice_position))

            # Do the fourier transform and add it to the I_aa matrix
            I_aa[z, :] += np.fft.fft(q_dot_lattice * positions[z].T[t_0:], n=fourier_length) \
                            .reshape((fourier_length,))[:int(fourier_length/2)]

        ic_sum = 0
        for tablename, table in timeseries.items():
            particle = particles.get_atom_from_tablename(tablename)
            if type(table) is np.ndarray:
                positions = table
            else:
                positions = np.array([
                    table.cols.pos_x,
                    table.cols.pos_y,
                    table.cols.pos_z
                ], dtype=np.complex128)

            ic_sum += cmath.exp(-1j * dot(q, particle.lattice_position)) * positions[z, t_0]

        I_aa[z, :] *= ic_sum
        I_aa[z, :] = np.power(np.abs(I_aa[z, :]), 2)
        # I_aa[z, :] = np.real(I_aa[z, :])

    frequencies = np.fft.fftfreq(fourier_length, o['dt'])[:int(fourier_length/2)]
    energies = c['Hz_to_meV'] * frequencies

    return I_aa, energies, frequencies
