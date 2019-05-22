# -*- coding: utf-8 -*-

import numpy as np
from utils import dot
import cmath

def transform_on_q(q, options, constants, timeseries, particles, cutoff = -1):
    o, c = options, constants
    if o['debug']:
        print('Computing intensities for q = [{}, {}, {}]'.format(q[0], q[1], q[2]))

    sum_A = [0, 0, 0]
    sum_B = [[], [], []]

    transformed = [[], [], []]

    frequencies = None
    energies = None

    cutoff_index = -1

    # Sum over each particle
    for tablename, table in timeseries.items():
        particle = particles.get_atom_from_tablename(tablename)
        positions = [
            table.cols.pos_x,
            table.cols.pos_y,
            table.cols.pos_z
        ]

        q_dot_lattice = cmath.exp(-1j * dot(q, particle.lattice_position))

        for z in range(0, 3):
            # Sum over scattering vector dotted the particles lattice position
            sum_A[z] += positions[z][0] * q_dot_lattice

            # Prepare data for fourier transform
            fft_data = np.tensordot(positions[z], q_dot_lattice, axes=0)

            # Pad the data with zeroes to speed up the transform
            fft_data = np.append(fft_data, np.zeros(2 ** (len(fft_data) - 1).bit_length() - len(fft_data)))

            # Execute the transform
            Y = np.abs(np.fft.fft(fft_data))

            # Calculate the frequencies and energies these intensities correspond to
            if frequencies is None:
                frequencies = np.fft.fftfreq(len(fft_data), o['dt'])
                energies = c['Hz_to_meV'] * frequencies

            # Calculate the intensities
            # if cutoff_index >:
            #     sampled = Y[:cutoff_index]
            # elif cutoff > 0:
            sampled = Y

            L = len(sampled)
            P2 = abs(sampled / L)
            fourier_temp = P2[:int(L / 2)]
            fourier_temp[1:] = 2 * fourier_temp[1:]

            transformed[z] = fourier_temp

        # Sum over the intensities for each particle.
        for z in range(0, 3):
            sum_length = len(sum_B[z])
            for idx in range(0, len(transformed[z])):
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

def transform_on_q_2(q, options, constants, timeseries, particles):
    o, c = options, constants
    if o['debug']:
        print('Computing intensities for q = [{}, {}, {}]'.format(q[0], q[1], q[2]))

    # Sets the length to the next power of 2, for speed
    fourier_length = int(2 ** np.ceil(np.log2(len(timeseries['p0'].cols.pos_x))))
    I_aa = np.zeros((3, int(fourier_length / 2)), dtype=np.dtype('complex128'))

    for z in range(0, 3):
        for tablename, table in timeseries.items():
            particle = particles.get_atom_from_tablename(tablename)
            positions = np.array([
                table.cols.pos_x,
                table.cols.pos_y,
                table.cols.pos_z
            ])

            q_dot_lattice = cmath.exp(-1j * dot(q, particle.lattice_position))

            # Do the fourier transform and add it to the I_aa matrix
            I_aa[z, :] += np.real(np.fft.fft(positions[z].T * q_dot_lattice, n=fourier_length)).reshape((fourier_length,))[:int(fourier_length/2)]

        ic_sum = 0
        for tablename, table in timeseries.items():
            particle = particles.get_atom_from_tablename(tablename)
            positions = np.array([
                table.cols.pos_x,
                table.cols.pos_y,
                table.cols.pos_z
            ])

            ic_sum += cmath.exp(-1j * dot(q, particle.lattice_position)) * positions[z, 0]

        I_aa[z, :] *= ic_sum

    frequencies = np.fft.fftfreq(fourier_length, o['dt'])[:int(fourier_length/2)]
    energies = c['Hz_to_meV'] * frequencies

    return I_aa, energies, frequencies
