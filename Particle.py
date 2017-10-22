# -*- coding: utf-8 -*-
"""
Class Particle wraps a OBAtom with a few extra goodies. Cleaner than extending swig++ classes.
"""

import numpy as np
import math
from utils import cross, dot


class Particle(object):
    def __init__(self, id, obatom, N, neighbours, options):
        self.id = id
        self.N = N
        self.B_eff = np.array([0.0, 0.0, 0.0], dtype='float')
        self.options = options
        self.neighbours = neighbours
        self.pos = np.array([obatom.GetX(), obatom.GetY(), obatom.GetZ()], dtype='float')
        self.pos = self.pos / math.sqrt(dot(self.pos, self.pos))

        x, y, z = self.pos[0], self.pos[1], self.pos[2]

        # Setup spherical
        self.r = 1 # We simulate the spin unit vector.
        self.theta = math.atan2(math.sqrt(x ** 2 + y ** 2), z)
        self.phi = math.atan2(y, x)

    # Skal være spin position, ikke lattice position. J tager højde for interaktioner.
    def current_position(self):
        return self.pos.copy()

    def set_position(self, p):
        self.pos = p

    # Calculate the effective B field for this atom from it's neighbours
    def combine_neighbours(self, neighbours):
        # Kun nærmeste! Interaktionen til genbo vil være meget lavere og dermed approximeret væk
        self.B_eff = np.copy(self.options.B)
        for item in self.neighbours:
            if item != self.id:
                self.B_eff += -2 * self.options.J * self.options.spin * neighbours[item].current_position() / \
                          (self.options.g * self.options.mu_b)

    # Actually calculate the step.
    def calculate_RK4_cartesian(self, p):
        o = self.options

        return o.gamma * o.spin * (cross(p, self.B_eff) + o.l * (self.B_eff * dot(p, p) - p * dot(p, self.B_eff)))

    def take_RK4_step(self, b_rand):
        # Get the current position
        p = self.current_position()

        # Calculate partial Runge Kutta steps
        k1 = self.calculate_RK4_cartesian(p)
        k2 = self.calculate_RK4_cartesian(p + k1 * self.options.dt / 2)
        k3 = self.calculate_RK4_cartesian(p + k2 * self.options.dt / 2)
        k4 = self.calculate_RK4_cartesian(p + k3 * self.options.dt)

        # Calculate the total difference in the spin
        d_spin = (k1 + 2 * k2 + 2 * k3 + k4) * self.options.dt / 6

        # Calculate the random energy added
        # This is based on temperature
        d_spin_rand = self.options.gamma * self.options.spin * cross(p, b_rand)

        # Calculate new position and normalise the vector
        p = p + d_spin + d_spin_rand
        p = p / math.sqrt(dot(p, p))

        # Save the data to the atom
        self.set_position(p)

        return (self.id, p)

    # Calculate energy (Sum over spins dotted on nearest neighbours, times J).
    def get_energy(self, neighbours):
        energy = 0
        for item in self.neighbours:
            # -1 måske?
            # Hamiltonian
            energy += -1 * self.options.J * dot(neighbours[item].current_position(), self.current_position())

        return energy
