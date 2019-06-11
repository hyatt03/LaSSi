# -*- coding: utf-8 -*-

import math
import sys
import random


def simulation_iterator(options, constants, particles, iterations, tables, i_0=0):
    o, c = options, constants
    sigma = math.sqrt(2. * o['l'] * c['k_b'] * o['T'] * c['hbar'] /
                      ((c['g'] * c['mu_b']) ** 2. * o['spin']))

    # Begin simulation
    perc = 0
    i = i_0
    # Use a while loop so we can run on conditionals depending on i_0
    while i <= iterations:
        progress = int(100 * i / iterations)
        if options['debug'] and progress > perc:
            perc = progress
            print('Simulating {0}%\r'.format(perc))
            sys.stdout.flush()

        # ensure the effective B field is correct.
        particles.combine_neighbours()

        # Create a random pertubation to emulate temperature
        b_rand = random.gauss(0, sigma)
        u = random.random() * 2 * math.pi
        v = math.acos(2 * random.random() - 1)
        b_rand_sph = (b_rand, u, v)

        # Evaluate each particle to compute the next state
        for particle in particles.atoms:
            tablename = 'p{}'.format(particle.id)

            # Take a step
            if o['integrator'] == 'ad_bs':
                # Adams Bashforth method, 5th order, both numerically and energetically stable, but not great accuracy.
                id, pos = particle.ad_bs_step(b_rand_sph)
            elif o['integrator'] == 'ad3':
                id, pos = particle.ad3_step(b_rand_sph)
            elif o['integrator'] == 'RK4':
                # Fourth order Runge Kutta, pretty common method, but not stable in energy
                id, pos = particle.take_rk4_step(b_rand_sph)
            elif o['integrator'] == 'RK2':
                # Also known as the midpoint method
                id, pos = particle.take_rk2_step(b_rand_sph)
            elif o['integrator'] == 'euler':
                id, pos = particle.take_euler_step(b_rand_sph)
            else:
                raise ValueError('Invalid integrator, use ad_bs or RK4 in simulation options')

            # Get the energy of the particle
            energy = particle.get_energy(particles.atoms)

            # Save the data to the buffer
            tables[tablename].append([(
                    i * o['dt'],  # t
                    pos[0],  # x
                    pos[1],  # y
                    pos[2],  # z
                    energy
            )])

            # Flush the data to file once in a while, increase to run faster (costs more memory)
            if i % 1e7 == 0:
                tables[tablename].flush()

        # Add to the counter
        i += 1

    # Flush at the end of the simulation
    for particle in particles.atoms:
        tables['p{}'.format(particle.id)].flush()

    return tables
