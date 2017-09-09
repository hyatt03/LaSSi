"""
Class Particle wraps a OBAtom with a few extra goodies. Cleaner than extending swig++ classes.
"""

import numpy as np


class Particle(object):
    def __init__(self, id, obatom, N, options):
        self.id = id
        self.atom = obatom
        self.N = N
        self.B_eff = np.array([0.0, 0.0, 0.0], dtype='float64')
        self.options = options

    def current_position(self):
        return np.array([self.atom.GetX(), self.atom.GetY(), self.atom.GetZ()], dtype='float64')

    def set_position(self, p):
        self.atom.SetVector(p[0], p[1], p[2])

    # Calculate the effective B field for this atom from it's neighbours
    def combine_neighbours(self, neighbours):
        self.B_eff = self.options.B
        for item in neighbours:
            self.B_eff += -2 * self.options.J * self.options.spin * item.current_position() / \
                          (self.options.g * self.options.mu_b)

    def calculate_RK4_cartesian(self, p):
        o = self.options

        return (o.gamma * o.spin * np.cross(p, self.B_eff) - o.l * o.gamma * o.spin * (
            p * np.dot(p, self.B_eff) - self.B_eff * np.dot(p, p)))

    def take_RK4_step(self, b_rand):
        # Get the current position from OBAtom
        p = self.current_position()

        # Calculate partial Runge Kutta steps
        k1 = self.calculate_RK4_cartesian(p)
        k2 = self.calculate_RK4_cartesian(p + k1 * self.options.dt / 2)
        k3 = self.calculate_RK4_cartesian(p + k2 * self.options.dt / 2)
        k4 = self.calculate_RK4_cartesian(p + k3 * self.options.dt)

        # Calculate the total difference in the spin
        d_spin = (k1 + 2 * k2 + 2 * k3 + k4) * self.options.dt / 6

        # Calculate the random energy added
        d_spin_rand = self.options.gamma * self.options.spin * np.cross(p, b_rand)

        #print(d_spin, d_spin_rand)

        # Calculate new position and normalise the vector
        p = p + d_spin + d_spin_rand

        print(p)

        #p = p / np.linalg.norm(p)

        # Save the data to the atom
        self.set_position(p)

        return (self.id, p)
