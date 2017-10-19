# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import random
import sys
from utils import downsample

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

timeseries = []

def simulation_iterator(options, particles):
    sigma = math.sqrt(2 * options.l * options.k_b * options.T * options.hbar * options.dt / \
          ((options.g * options.mu_b) ** 2 * options.spin))

    # Begin simulation
    perc = 0
    for i in range(1, options.N_simulation + 1):
        if (100 * i) / options.N_simulation > perc:
            perc = (100 * i) / options.N_simulation
            print 'Simulating {0}%\r'.format(perc),
            sys.stdout.flush()

        # ensure the effective B field is correct.
        particles.combine_neighbours()

        b_rand = random.gauss(0, sigma)
        u = random.random() * 2 * math.pi
        v = math.acos(2 * random.random() - 1)
        b_rand_vec = b_rand * np.array([math.sin(v) * math.cos(u), math.sin(v) * math.sin(u), math.cos(v)])

        for particle in particles.atoms:
            id, pos = particle.take_RK4_step(b_rand_vec)
            energy = particle.get_energy(particles.atoms)
            timeseries.append((i * options.dt, id, pos[0], pos[1], pos[2], energy))

    return pd.DataFrame(timeseries, columns=('t', 'id', 'pos_x', 'pos_y', 'pos_z', 'energy'))

def plot_spins(results, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot(downsample(results['pos_x'], 1000), downsample(results['pos_y'], 1000), downsample(results['pos_z'], 1000))
    ax.plot(results['pos_x'], results['pos_y'], results['pos_z'])

    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    # plt.zlim(-1, 1)

    print('Saving spins plot')
    plt.savefig(filename, bbox_inches = 'tight')
