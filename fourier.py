# -*- coding: utf-8 -*-

import scipy.fftpack as sci
from scipy import linspace
import matplotlib.pyplot as plt
import numpy as np
from utils import dot
import cmath
import math
from operator import mul
from multiprocessing import Pool


def fourier(options, timeseries, particles):
    return run_computation_on_q([0, options, timeseries, particles])


def run_computation_on_q(arg):
    q_size, options, timeseries, particles = arg[0], arg[1], arg[2], arg[3]
    q = np.copy(options.q) * q_size

    sum_A = [0, 0, 0]
    sum_B = [[], [], []]

    frequencies = []
    energies = []

    for particle in particles:
        particle_time = timeseries.query('id == {}'.format(particle.id))
        position_t = [
            particle_time.pos_x.as_matrix(),
            particle_time.pos_y.as_matrix(),
            particle_time.pos_z.as_matrix()
        ]

        transformed = [[], [], []]

        for z in range(0, 3):
            sum_A[z] += position_t[z][0] * cmath.exp(-1j * dot(q, particle.lattice_position))

            fft_data = [x * cmath.exp(-1j * dot(q, particle.lattice_position)) for x in position_t[z]]

            while len(fft_data) < (2 ** (len(fft_data) - 1).bit_length()):
                fft_data.append(0)

            Y = sci.fft(fft_data)

            L = len(fft_data)
            P2 = abs(Y / L)
            fourier_temp = P2[:(L / 2)]
            fourier_temp[1:] = 2 * fourier_temp[1:]

            L = float(L)
            frequency = [float(x) / (L * options.dt) for x in np.arange(0, L/2 - 1, (L - 1) / L)]
            energy = [x * options.GHz_to_meV * 1e-9 for x in frequency]

            transformed[z] = fourier_temp
            energies = energy
            frequencies = frequency

            sum_length = len(sum_B[z])
            for idx in range(0, len(fourier_temp)):
                if sum_length <= idx:
                    sum_B[z].append(0)

                sum_B[z][idx] += transformed[z][idx]

    I_aa_temp = [
        [],  # xx
        [],  # yy
        []   # zz
    ]

    for i in range(0, 3):
        I_aa_temp[i] = sum_A[i] * np.array(sum_B[i])

    return [I_aa_temp, energies, frequencies]


def parallel_compute_scattering_intensity(options, timeseries, particles):
    qs = np.arange(0, 4, 0.01)
    arguments = []

    for q in qs:
        arguments.append((q, options, timeseries, particles))

    pool = Pool(processes=3)
    mapping_result = pool.map(run_computation_on_q, arguments)
    pool.close()
    pool.join()

    # q sizes, I_a_a_temp, energies, frequencies
    return qs, mapping_result[0], mapping_result[1], mapping_result[2]


def calculate_scattering_intensity(options, timeseries, particles):
    qs = np.arange(0, 4, 0.01)
    I_aa_temps = []
    energies = []
    frequencies = []

    for q_size in qs:
        I_aa_temp, energy, frequency = run_computation_on_q([q_size, options, timeseries, particles])
        I_aa_temps.append(I_aa_temp)
        energies.append(energy)
        frequencies.append(frequency)

    return qs, I_aa_temps, energies, frequencies


def plot_fourier(o, filename, total_fourier, f, energy, smoothing = False):
    y_data = (total_fourier[0][0:len(energy)] + total_fourier[1][0:len(energy)] + total_fourier[2][0:len(energy)]) / 3

    if smoothing:
        mu = 0
        sig = 0.005
        x = np.linspace(-500, 500, num = len(y_data))
        gauss = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        y_data = np.convolve(y_data, gauss, mode='same')
        y_data = -y_data
        a_max = np.amax(y_data[10:-1])
        y_data = y_data / a_max

    fig, ax = plt.subplots()
    ax.plot(energy, y_data)

    plt.xlim(0, 0.5)
    #plt.ylim(0, 1.5)
    plt.xlabel('Energy [meV]')
    plt.ylabel('Intensity [A.U.]')

    #plt.axis('equal')

    plt.savefig(filename, bbox_inches='tight', dpi=300)


def plot_energy_spectrum(o, total_fourier, f, energy):

    pass

