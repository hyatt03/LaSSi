# -*- coding: utf-8 -*-
"""
Class Particle wraps a OBAtom with a few extra goodies. Cleaner than extending swig++ classes.
"""

import numpy as np
import math
from utils import dot, to_sph, to_cart


class Particle(object):
    def __init__(self, id, atom, N, neighbours, options, constants):
        self.id = id
        self.N = N
        self.B_eff = np.array([0.0, 0.0, 0.0], dtype='float')
        self.B_eff_sph = 0., 0., 0.
        self.options = options
        self.constants = constants
        self.neighbours = neighbours
        self.lattice_position = atom.position

        r, theta, phi = to_sph(self.lattice_position)

        # Find initial position (just start with the lattice position)
        # Set r = 1 as we work with a unit vector representing the spin
        self.r = 1
        self.theta = theta
        self.phi = phi

        # We use cartesian when calculating the effective B-field
        self.pos = to_cart([self.r, self.theta, self.phi])

    # Skal være spin position, ikke lattice position. J tager højde for interaktioner.
    def current_position(self):
        return self.pos.copy()

    def set_position(self, theta, phi):
        self.theta = theta
        self.phi = phi
        self.pos = to_cart([1, theta, phi])

        return self.pos

    # Calculate the effective B field for this atom from it's nearest neighbours
    def combine_neighbours(self, neighbours):
        o, c = self.options, self.constants
        self.B_eff = np.copy(self.options['B'])
        for item in self.neighbours:
            if item != self.id:
                self.B_eff += -2 * o['J'] * o['spin'] * neighbours[item].current_position() / \
                          (c['g'] * c['mu_b'])

    def calculate_RK4(self, theta, phi):
        o, c, B = self.options, self.constants, self.B_eff
        d_theta = ((math.cos(theta) * B[0] * o['l'] - B[1]) * math.cos(phi) + (math.cos(theta) * B[1] * o['l'] + B[0]) *
                   math.sin(phi) - B[2] * math.sin(theta) * o['l']) * c['gamma']

        d_phi = ((B[0] * math.cos(phi) + B[1] * math.sin(phi)) * math.cos(theta) - B[2] * math.sin(theta) - o['l'] * (
                    B[0] * math.sin(phi) - B[1] * math.cos(phi))) * c['gamma'] / math.sin(theta)

        return d_theta, d_phi

    def take_RK4_step(self, b_rand):
        # Get the current position
        o, c = self.options, self.constants
        theta, phi = self.theta, self.phi
        b_mag, b_theta, b_phi = math.fabs(b_rand[0]), b_rand[1], b_rand[2]

        # Calculate partial Runge Kutta steps
        d_theta1, d_phi1 = self.calculate_RK4(theta, phi)
        d_theta2, d_phi2 = self.calculate_RK4(theta + d_theta1 * o['dt'] / 2, phi + d_phi1 * o['dt'] / 2)
        d_theta3, d_phi3 = self.calculate_RK4(theta + d_theta2 * o['dt'] / 2, phi + d_phi2 * o['dt'] / 2)
        d_theta4, d_phi4 = self.calculate_RK4(theta + d_theta3 * o['dt'], phi + d_phi3 * o['dt'])

        # Calculate the total difference in the spin
        d_theta = (d_theta1 + 2 * d_theta2 + 2 * d_theta3 + d_theta4) * o['dt'] / 6
        d_phi = (d_phi1 + 2 * d_phi2 + 2 * d_phi3 + d_phi4) * o['dt'] / 6

        # Calculate the random energy added
        # This is based on temperature
        d_spin_rand_theta = (-math.sin(b_theta) * math.sin(b_phi) * math.cos(phi) +
                             math.sin(b_theta) * math.cos(b_phi) * math.sin(phi)) * c['gamma'] * b_mag
        d_spin_rand_phi = ((math.sin(phi) * math.sin(b_theta) * math.sin(b_phi) +
                            math.sin(b_theta) * math.cos(b_phi) * math.cos(phi)) * math.cos(theta) -
                           math.cos(b_theta) * math.sin(theta)) * c['gamma'] * b_mag / math.sin(theta)

        # Calculate the final position
        theta += d_theta + d_spin_rand_theta
        phi += d_phi + d_spin_rand_phi

        # Save the data to the atom
        p = self.set_position(theta, phi)

        return (self.id, p)

    # Calculate energy using hamiltonian
    def get_energy(self, neighbours):
        energy = 0
        for item in self.neighbours:
            if item != self.id:
                # Hamiltonian
                energy += -2 * self.options['J'] * dot(neighbours[item].current_position(), self.current_position())

        return energy
