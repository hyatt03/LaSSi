# -*- coding: utf-8 -*-

import math
import numpy as np
from numba import jit

import matplotlib.pyplot as plt

import vectorized_integrators.vectorized_adams_bashforth as ad_bs


@jit(nopython=True)
def compute_effective_magnetic_fields(positions, neighbours, b_fields_0, b_eff_p, anisotropies, enable_anisotropies):
    b_eff = np.copy(b_fields_0)

    for p_id in range(len(positions)):
        if enable_anisotropies:
            # Compute anisotropy field
            b_eff[p_id, :] -= np.dot(anisotropies[p_id], positions[p_id, :])

        # Iterate through the neighbours
        for n_id in neighbours[p_id]:
            if n_id != p_id and n_id != -1:
                # Add the the effective field from
                b_eff[p_id, :] += b_eff_p * positions[n_id, :]

    return b_eff


@jit(nopython=True)
def take_sim_steps(iterations, i_0, debug, positions, neighbours, b_fields_0, b_eff_p,
                   anisotropies, enable_anisotropies, sigma, n_particles, gamma, lam, dt, spin):
    # Initialize array to contain data
    data = np.zeros((positions.shape[0], positions.shape[1] + 2, int(iterations) + 1))

    # Initialize arrays to hole intermediary values
    sx = np.empty((4, positions.shape[0]))
    sx[:, :] = np.nan
    sy = np.empty((4, positions.shape[0]))
    sy[:, :] = np.nan
    sz = np.empty((4, positions.shape[0]))
    sz[:, :] = np.nan

    # Begin simulation
    perc = 0
    i = i_0

    # Use a while loop so we can run on conditionals depending on i_0
    while i <= iterations:
        # Compute the progress
        progress = int(100 * i / iterations)
        if debug and progress > perc:
            perc = progress
            print('Simulating', perc, '%')

        # ensure the effective B field is correct.
        b_eff = compute_effective_magnetic_fields(positions, neighbours, b_fields_0, b_eff_p,
                                                  anisotropies, enable_anisotropies)
        b_eff_x, b_eff_y, b_eff_z = b_eff[:, 0], b_eff[:, 1], b_eff[:, 2]

        # Create a random pertubation to emulate temperature
        b_rands_mag = np.random.normal(0, sigma, size=n_particles)
        us = np.random.uniform(0, 1, n_particles) * 2 * np.pi
        vs = np.arccos(2 * np.random.uniform(0, 1, n_particles) - 1)
        b_rands_x = b_rands_mag * np.sin(vs) * np.cos(us)
        b_rands_y = b_rands_mag * np.sin(vs) * np.sin(us)
        b_rands_z = b_rands_mag * np.cos(vs)

        # Grab x, y, and z from positions
        p_x, p_y, p_z = positions[:, 0], positions[:, 1], positions[:, 2]

        # Take a step
        p_x, p_y, p_z, sx, sy, sz = ad_bs.ad_bs_step(p_x, p_y, p_z, b_eff_x, b_eff_y, b_eff_z,
                                                     b_rands_x, b_rands_y, b_rands_z, lam, gamma, dt, spin, sx, sy, sz)

        # Compute new positions based on the derivative
        positions[:, 0] = p_x
        positions[:, 1] = p_y
        positions[:, 2] = p_z

        # Normalize the coordinates
        for p_id in range(n_particles):
            norm = np.sqrt(np.sum(positions[p_id, :] * positions[p_id, :]))
            positions[p_id, :] = positions[p_id, :] / norm

        # Save the current position and advance i
        data[:, 1:4, i] = np.copy(positions)
        data[:, 0, i] = i * dt
        data[:, -1, i] = 0.0

        i += 1

    # Save the data
    return data


def vectorized_simulation_iterator(options, constants, particles, iterations, tables, i_0=0):
    o, c = options, constants

    # Compute temperature sigma and the effective B field parameter
    b_eff_p = -2 * o['J'] * o['spin'] / (c['g'] * c['mu_b'])
    sigma = math.sqrt(2. * o['l'] * c['k_b'] * o['T'] * c['hbar'] * o['dt'] /
                      ((c['g'] * c['mu_b']) ** 2. * o['spin']))

    # Create an array of positions and effective B fields
    n_particles = len(particles.atoms)
    positions = np.zeros((n_particles, 3))
    b_fields_0 = np.empty_like(positions)

    # Check what we want to enable
    enable_anisotropies = o['anisotropy'] is not None

    # Create a couple of lists containing the couplings
    neighbours = []
    anisotropies = []

    # Find any given particles couplings
    max_neighbour_size = 0
    for row_i, particle in enumerate(particles.atoms):
        # Find the neighbours
        if len(particle.neighbours) == 0:
            neighbours.append([particle.id])
        else:
            neighbours.append(particle.neighbours)

        # Save the largest size to avoid reflected lists
        if len(neighbours[-1]) > max_neighbour_size:
            max_neighbour_size = len(neighbours[-1])

        # Check for anisotropies
        if enable_anisotropies:
            anisotropies.append(np.copy(o['anisotropy'][particle.id, :, :]))
        else:
            anisotropies.append(np.zeros(3))

        # Set the initial field
        b_fields_0[row_i, :] = np.copy(o['B'])

        # Set the initial position
        positions[row_i, :] = particle.current_position()

    # Convert anisotropies to numpy array
    anisotropies = np.array(anisotropies)

    # Convert neighbours to numpy array
    np_neighbours = np.zeros((len(neighbours), max_neighbour_size), dtype=int) - 1
    for idx, neighbours_set in enumerate(neighbours):
        for idx_2, n_id in enumerate(neighbours_set):
            np_neighbours[idx, idx_2] = n_id

    # Run a simulation
    data = take_sim_steps(iterations, i_0, options['debug'], positions, np_neighbours, b_fields_0, b_eff_p,
                          anisotropies, enable_anisotropies, sigma, n_particles, c['gamma'], o['l'], o['dt'], o['spin'])

    plt.plot(data[1, 0, :], data[1, 1, :])
    plt.plot(data[1, 0, :], data[1, 2, :])
    plt.plot(data[1, 0, :], data[1, 3, :])
    plt.show()

    # Save data to table
    for p_id in range(n_particles):
        tablename = 'p' + str(p_id)
        tables[tablename].append(data[p_id, :, :].T)
        tables[tablename].flush()
