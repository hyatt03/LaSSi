# -*- coding: utf-8 -*-

import math
import numpy as np
import random

timeseries = []

def simulation_iterator(options, particles):
    # Create a suitable filename
    filename = 'Particles={}_N={}_dt={}_l={}_T={}_B={}_Particle_' \
        .format(particles.N_atoms, options.N_simulation, options.dt, options.l, options.T, options.B)

    perc = 0

    # Begin simulation
    for i in range(1, options.N_simulation):
        if (100 * i) / options.N_simulation > perc:
            perc = (100 * i) / options.N_simulation
            print(str(perc) + '%')

        # ensure the effective B field is correct.
        particles.combine_neighbours()

        var = 2 * options.l * options.k_b * options.T * options.hbar * options.dt / \
              ((options.g * options.mu_b) ** 2 * options.spin)

        # Spørgsmål til Tang: Skal der tilfældighed for hver iteration, eller hver partikel ved hver iteration?
        # Lille pertubation til B feltet (der kommer noget energi i systemet)
        b_rand = random.gauss(0, (var) ** (1 / 2)) * options.dt
        u = random.random() * 2 * math.pi
        v = math.acos(2 * random.random() - 1)
        b_rand_vec = b_rand * np.array([math.sin(v) * math.cos(u), math.sin(v) * math.sin(u), math.cos(v)])

        for particle in particles:
            id, pos = particle.take_RK4_step(b_rand_vec)
            timeseries.append((i * options.dt, id, pos))

    return timeseries

