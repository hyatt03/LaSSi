# -*- coding: utf-8 - *-

import random
import math
import numpy as np

def anneal_particles(options, particles):
    T_stair_steps = np.linspace(options.T, options.T * 10, num=options.anneal)[::-1]

    for T in T_stair_steps:
        options.T = T
        particles.combine_neighbours()

        var = 2 * options.l * options.k_b * T * options.hbar * options.dt / \
              ((options.g * options.mu_b) ** 2 * options.spin)

        b_rand = random.gauss(0, (var) ** (1 / 2)) * options.dt
        u = random.random() * 2 * math.pi
        v = math.acos(2 * random.random() - 1)
        b_rand_vec = b_rand * np.array([math.sin(v) * math.cos(u), math.sin(v) * math.sin(u), math.cos(v)])

        for particle in particles:
            particle.take_RK4_step(b_rand_vec)

    return particles
