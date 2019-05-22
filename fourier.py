# -*- coding: utf-8 -*-

import numpy as np
from utils import dot
import cmath


def transform_on_q(q, options, constants, timeseries, particles):
    o, c = options, constants
    if o['debug']:
        print('Computing intensities for q = [{}, {}, {}]'.format(q[0], q[1], q[2]))

    sum_A = [0, 0, 0]
    sum_B = [[], [], []]

    frequencies = []
    energies = []

    # Sum over each particle
    for tablename, table in timeseries.items():
        particle = particles.get_atom_from_tablename(tablename)
        positions = [
            table.cols.pos_x,
            table.cols.pos_y,
            table.cols.pos_z
        ]

        transformed = [[], [], []]
        q_dot_lattice = cmath.exp(-1j * dot(q, particle.lattice_position))

        for z in range(0, 3):
            # Sum over scattering vector dotted the particles lattice position
            sum_A[z] += positions[z][0] * q_dot_lattice

            # Prepare data for fourier transform
            fft_data = [x * q_dot_lattice for x in positions[z]]

            # Pad the data with zeroes to speed up the transform
            while len(fft_data) < (2 ** (len(fft_data) - 1).bit_length()):
                fft_data.append(0)

            # Execute the transform
            Y = np.abs(np.fft.fft(fft_data))

            # Calculate the intensities
            # sampled = downsample(Y, 50000)
            sampled = Y

            L = len(sampled)
            P2 = abs(sampled / L)
            fourier_temp = P2[:int(L / 2)]
            fourier_temp[1:] = 2 * fourier_temp[1:]

            # Calculate the frequencies and energies these intensities correspond to
            frequency = np.fft.fftfreq(len(fft_data), o['dt'])
            energy = c['Hz_to_meV'] * frequency

            transformed[z] = fourier_temp
            energies = energy
            frequencies = frequency

            # Sum over the intensities for each particle.
            sum_length = len(sum_B[z])
            for idx in range(0, len(fourier_temp)):
                if sum_length <= idx:
                    sum_B[z].append(0)

                sum_B[z][idx] += transformed[z][idx]

    I_aa_temp = [
        [],  # xx
        [],  # yy
        []  # zz
    ]

    # Multiply the two components of the scattering intensities to compute the final intensity
    for i in range(0, 3):
        I_aa_temp[i] = sum_A[i] * np.array(sum_B[i])

    return [np.array(I_aa_temp), energies, frequencies]
