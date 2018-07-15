# -*- coding: utf-8 - *-

import random
import math
import numpy as np
from multiprocessing import Pool


def run_anneal(arg):
    return anneal_particles(arg[0], arg[1])


def parrallel_anneal(options, particles, sets):
    particle_sets = []
    for i in range(0, sets):
        particle_sets.append((options, particles))

    pool = Pool()
    mapping_result = pool.map(run_anneal, particle_sets)
    pool.close()
    pool.join()

    lowest_energy = None
    best_set = None
    for particle_set in mapping_result:
        e = particle_set.get_energy()

        if not lowest_energy or e < lowest_energy:
            lowest_energy = e
            best_set = particle_set

    return best_set


def anneal_particles(options, constants, particles, steps):
    T_stair_steps = np.linspace(0, options['T'] * 10, num = steps)[::-1]

    for T in T_stair_steps:
        options['T'] = T
        particles.combine_neighbours()

        var = 2 * options['l'] * constants['k_b'] * T * constants['hbar'] * options['dt'] / \
              ((constants['g'] * constants['mu_b']) ** 2 * options['spin'])

        b_rand = random.gauss(0, math.sqrt(var)) * options['dt']
        u = random.random() * 2 * math.pi
        v = math.acos(2 * random.random() - 1)
        b_rand_sph = (b_rand, u, v)

        for particle in particles:
            particle.take_RK4_step(b_rand_sph)

    return particles
