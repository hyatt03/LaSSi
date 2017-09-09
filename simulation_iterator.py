# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 13:09:09 2017

@author: Rasmus
"""
import csv
import math
import numpy as np

import random

# from load_simulation import load_simulation
# from particle_cartesian import Particle
# from initialize import initialize_particles
# from simulation import simulation

# Unused imports
"""
from savedata import savedata
from realspace import realspace_data
from list_linspace import list_linspace
from realspace_plot import realspace_plot
from fourier import fourier
from combine_fourier import combine_fourier
from only_fourier_fnct import only_fourier_fnct
from only_fourier_plot_fnct import only_fourier_plot_fnct
from I_q_w import I_q_omega
import matplotlib.pyplot as plt
import datetime as date
from RK4 import RK4
"""

timeseries = []

def simulation_iterator(options, particles):
    # Create a suitable filename
    filename = 'Particles={}_N={}_dt={}_l={}_T={}_B={}_Particle_' \
        .format(particles.N_atoms, options.N_simulation, options.dt, options.l, options.T, options.B)

    # Begin simulation
    for i in range(1, options.N_simulation):
        # ensure the effective B field is correct.
        print('qqq')
        particles.combine_neighbours()
        print('qqq2')

        var = 2 * options.l * options.k_b * options.T * options.hbar * options.dt / \
              ((options.g * options.mu_b) ** 2 * options.spin)

        # Spørgsmål til Tang: Skal der tilfældighed for hver iteration, eller hver partikel ved hver iteration?
        # Lille pertubation til B feltet (der kommer noget energi i systemet)
        b_rand = random.gauss(0, (var) ** (1 / 2))
        u = random.random() * 2 * math.pi
        v = math.acos(2 * random.random() - 1)
        b_rand_vec = b_rand * np.array([math.sin(v) * math.cos(u), math.sin(v) * math.sin(u), math.cos(v)])

        for particle in particles:
            print(particle)
            id, pos = particle.take_RK4_step(b_rand_vec)
            timeseries.append((i * options.dt, id, pos))


    """
    Creating particles and giving neighbours and IC
    """

    # particle_list = initialize_particles(N_particles, N_neighbours, triangle, chain, chain_in_the_middle, hourglass,
    #                                     ferro, J, spin, B_ext)

    """
    Simulation and data saving
    """
    # particle_list, qqq = simulation(particle_list, N_simulation, dt, B_ext, spin, J, T, l, filename, directory, split, anneal)


    """
    if angles == 1:

        positions = load_simulation(directory + "Kode/Data/Realspace/", "split*" + filename + "*.csv", N_particles,
                                    split)
        angles = np.zeros(len(positions[0][0]), )

        for i in range(len(angles)):
            angles[i] = math.acos(
                positions[0][0][i] * positions[1][0][i] + positions[0][1][i] * positions[1][1][i] + positions[0][2][i] *
                positions[1][2][i])

        with open((directory + "/Kode/Data/Angles/Angles_" + filename.replace("_Particle_", "")).replace(".",
                                                                                                         "_") + ".csv",
                  'w', newline='') as csvfile:
            writer = csv.writer(csvfile, lineterminator="\n")
            writer.writerows(zip(*[angles]))

    Particle.particle_count = 0
    """

    # return qqq
