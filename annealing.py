# -*- coding: utf-8 - *-

import random
import math
import numpy as np


def anneal_particles(options, constants, particles, steps):
    prior_temperature = options['T']
    anneal_T = options['T']

    if options['anneal_T']:
        # Necessary to start at much higher temperature when not using the optional parameter anneal_T
        anneal_T = options['anneal_T'] / 10

    # Creates an array of temperatures that we want to anneal through
    T_stair_steps = np.linspace(anneal_T * 10, 0, num = steps)

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

    options['T'] = prior_temperature

    return particles
