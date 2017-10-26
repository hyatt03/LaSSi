# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import random
import sys

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
            # print('Size of timeseries: {}, at {} iterations'.format(sys.getsizeof(timeseries), i))
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
    print('Saving spins plot')

    # 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(results['pos_x'], results['pos_y'], results['pos_z'])

    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    ax.set_xlabel('$S_x$')
    ax.set_ylabel('$S_y$')
    ax.set_zlabel('$S_z$')

    plt.axis('equal')
    plt.title('3d projektion af spin af enkelt Gd ion')

    plt.savefig('{}_3d.png'.format(filename), bbox_inches = 'tight', dpi=300)

    # xy projection
    fig, ax = plt.subplots()
    ax.plot(results['pos_x'], results['pos_y'])

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.xlabel('$S_x$')
    plt.ylabel('$S_y$')

    plt.axis('equal')
    plt.title('Projektion af xy planet for spin af enkelt Gd ion')

    plt.savefig('{}_xy.png'.format(filename), bbox_inches='tight', dpi=300)
