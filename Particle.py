# -*- coding: utf-8 -*-
"""
Class Particle wraps a OBAtom with a few extra goodies. Cleaner than extending swig++ classes.
"""

import numpy as np
import math
from math import sin, cos
from utils import dot, to_sph, to_cart

ab = [1901 / 720, 2774 / 720, 2616 / 720, 1274 / 720, 251 / 720]


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

        # We calculate a constant once, so we don't have to do it for each iteration
        self.b_eff_p = -2 * options['J'] * options['spin'] / (constants['g'] * constants['mu_b'])

        # We set the previous steps for use with Adams Bashforth integration
        self.sphi4, self.sphi3, self.sphi2, self.sphi1 = None, None, None, None
        self.stheta4, self.stheta3, self.stheta2, self.stheta1 = None, None, None, None

    # Skal være spin position, ikke lattice position. J tager højde for interaktioner.
    def current_position(self):
        return self.pos.copy()

    # Sets the position in memory
    def set_position(self, theta, phi):
        self.theta = theta
        self.phi = phi
        self.pos = to_cart([1, theta, phi])

        return self.pos

    # Calculate the effective B field for this atom from it's nearest neighbours
    def combine_neighbours(self, neighbours):
        o, c = self.options, self.constants
        # We start with a copy of the external field
        self.B_eff = np.copy(o['B'])

        # Iterate over the neighbours to find the effective B field
        for item in self.neighbours:
            if item != self.id:
                self.B_eff += self.b_eff_p * neighbours[item].current_position()

    # Evaluate the differential equation given a specific theta, phi and random field.
    def calculate_function_value(self, theta, phi, b_rand=None):
        o, c, B = self.options, self.constants, self.B_eff

        # Check if a random field is passed in, only works using linear methods
        # When using Runge-Kutta, we integrate this seperately
        if b_rand is not None:
            b, u, v = b_rand
        else:
            b, u, v = 0, 0, 0

        # Handle edge case
        try:
            stheta = 1 / sin(theta)
        except ZeroDivisionError:
            stheta = 1 / 1e-20  # Small number

        try:
            cphi = 1 / cos(phi)
        except ZeroDivisionError:
            cphi = 1 / 1e-20

        stheta_cphi = stheta * cphi

        # Evaluate the difference in theta
        d_theta = stheta * c['gamma'] * sin(theta) * ((-b * sin(u) * sin(v) + B[0] * o['l'] * cos(theta) - B[1]) * cos(phi)
                                               + (b * sin(u) * cos(v) + B[1] * o['l'] * cos(theta) + B[0]) * sin(phi)
                                               - B[2] * sin(theta) * o['l'])

        # Evaluate the difference in phi
        d_phi = c['gamma'] * stheta_cphi * (
            + B[0] * o['l'] * cos(phi) * sin(phi) * (cos(theta) ** 2.)
            - B[1] * o['l'] * (cos(phi) ** 2.) * (cos(theta) ** 2.)
            - B[2] * o['l'] * cos(theta) * sin(phi) * sin(theta)
            - B[0] * o['l'] * cos(phi) * sin(phi)
            + B[1] * o['l'] * (cos(phi) ** 2.)
            + B[1] * o['l'] * (cos(theta) ** 2.)
            - b * cos(u) * cos(phi) * sin(theta)
            + b * cos(theta) * sin(u) * cos(v)
            - B[2] * cos(phi) * sin(theta)
            + B[0] * cos(theta)
            - (d_theta * sin(phi) * cos(theta) / c['gamma'])
        )

        return d_theta, d_phi

    # Evaluate the fifth step using function values from the past 4 steps
    def fifth_ad_bs_step(self, theta, phi, b_rand):
        d_ft, d_fp = self.calculate_function_value(theta, phi, b_rand)

        # New weighted d_stheta and d_sphi
        dt = ab[0] * d_ft - ab[1] * self.stheta4 + ab[2] * self.stheta3 - ab[3] * self.stheta2 + ab[4] * self.stheta1
        dp = ab[0] * d_fp - ab[1] * self.sphi4 + ab[2] * self.sphi3 - ab[3] * self.sphi2 + ab[4] * self.sphi1

        # Return d_stheta, d_sphi as well as the function values
        return dt, dp, d_ft, d_fp

    # Evaluate the fourth step using function values from the past 3 steps
    def fourth_ad_bs_step(self, theta, phi, b_rand):
        d_ftheta, d_fphi = self.calculate_function_value(theta, phi, b_rand)
        return (55 / 24) * d_ftheta - (59 / 24) * self.stheta4 + (37 / 24) * self.stheta3 - (9 / 24) * self.stheta2, \
               (55 / 24) * d_fphi - (59 / 24) * self.sphi4 + (37 / 24) * self.sphi3 - (9 / 24) * self.sphi2, \
               d_ftheta, \
               d_fphi

    # Evaluate the third step using function values from the past 2 steps
    def third_ad_bs_step(self, theta, phi, b_rand):
        d_ftheta, d_fphi = self.calculate_function_value(theta, phi, b_rand)
        return (23 / 12) * d_ftheta - (16 / 12) * self.stheta4 + (5 / 12) * self.stheta3, \
               (23 / 12) * d_fphi - (16 / 12) * self.sphi4 + (5 / 12) * self.sphi3, \
               d_ftheta, \
               d_fphi

    # Evaluate the second step using function values from the past step
    def second_ad_bs_step(self, theta, phi, b_rand):
        d_ftheta, d_fphi = self.calculate_function_value(theta, phi, b_rand)
        return (3 / 2) * d_ftheta - (1 / 2) * self.stheta4, \
               (3 / 2) * d_fphi - (1 / 2) * self.sphi4, \
               d_ftheta, \
               d_fphi

    # Evaluate the first step (basically forward Euler)
    def first_ad_bs_step(self, theta, phi, b_rand):
        return self.calculate_function_value(theta, phi, b_rand)

    # Integrate one step using third order Adams Bashforth
    # This is a linear multistep method (https://en.wikipedia.org/wiki/Linear_multistep_method)
    def ad3_step(self, b_rand):
        # Grab the constants
        o, c = self.options, self.constants
        theta, phi = self.theta, self.phi

        # Gradually increase steps in the Adams Bashforth method
        if self.sphi4 is None:
            d_stheta, d_sphi = self.first_ad_bs_step(theta, phi, b_rand)
            d_ftheta, d_fphi = d_stheta, d_sphi
        elif self.sphi3 is None:
            d_stheta, d_sphi, d_ftheta, d_fphi = self.second_ad_bs_step(theta, phi, b_rand)
        else:
            d_stheta, d_sphi, d_ftheta, d_fphi = self.third_ad_bs_step(theta, phi, b_rand)

        # Move the values along so we keep continuing
        self.stheta4, self.stheta3, self.stheta2, self.stheta1 = d_ftheta, self.stheta4, self.stheta3, self.stheta2
        self.sphi4, self.sphi3, self.sphi2, self.sphi1 = d_fphi, self.sphi4, self.sphi3, self.sphi2

        # Take the step and calculate the final position
        theta += d_stheta * o['dt']
        phi += d_sphi * o['dt']

        # Save the data to the atom
        p = self.set_position(theta, phi)

        return self.id, p

    # Integrate one step using fifth order Adams Bashforth
    # This is a linear multistep method (https://en.wikipedia.org/wiki/Linear_multistep_method)
    def ad_bs_step(self, b_rand):
        # Grab the constants
        o, c = self.options, self.constants
        theta, phi = self.theta, self.phi

        # Gradually increase steps in the Adams Bashforth method
        if self.sphi4 is None:
            d_stheta, d_sphi = self.first_ad_bs_step(theta, phi, b_rand)
            d_ftheta, d_fphi = d_stheta, d_sphi
        elif self.sphi3 is None:
            d_stheta, d_sphi, d_ftheta, d_fphi = self.second_ad_bs_step(theta, phi, b_rand)
        elif self.sphi2 is None:
            d_stheta, d_sphi, d_ftheta, d_fphi = self.third_ad_bs_step(theta, phi, b_rand)
        elif self.sphi1 is None:
            d_stheta, d_sphi, d_ftheta, d_fphi = self.fourth_ad_bs_step(theta, phi, b_rand)
        else:
            d_stheta, d_sphi, d_ftheta, d_fphi = self.fifth_ad_bs_step(theta, phi, b_rand)

        # Move the values along so we keep continuing
        self.stheta4, self.stheta3, self.stheta2, self.stheta1 = d_ftheta, self.stheta4, self.stheta3, self.stheta2
        self.sphi4, self.sphi3, self.sphi2, self.sphi1 = d_fphi, self.sphi4, self.sphi3, self.sphi2

        # Take the step and calculate the final position
        theta += d_stheta * o['dt']
        phi += d_sphi * o['dt']

        # Save the data to the atom
        p = self.set_position(theta, phi)

        return self.id, p

    # Integrate one step using second order Runge-Kutta (https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
    # This is a nonlinear method as it using partial time steps
    # This integrator does not have energy conservation for some systems
    def take_rk2_step(self, b_rand):
        # Get the current position
        o, c = self.options, self.constants
        theta, phi = self.theta, self.phi
        b_mag, b_theta, b_phi = math.fabs(b_rand[0]), b_rand[1], b_rand[2]

        # Calculate partial Runge Kutta steps
        d_theta1, d_phi1 = self.calculate_function_value(theta, phi)
        d_theta2, d_phi2 = self.calculate_function_value(theta + d_theta1 * o['dt'] / 2, phi + d_phi1 * o['dt'] / 2)

        # Calculate the total difference in the spin
        d_theta = d_theta2 * o['dt']
        d_phi = d_phi2 * o['dt']

        # Calculate the random energy added
        # This is based on temperature
        d_spin_rand_theta = (-math.sin(b_theta) * math.sin(b_phi) * math.cos(phi) +
                             math.sin(b_theta) * math.cos(b_phi) * math.sin(phi)) * c['gamma'] * b_mag * o['dt']

        d_spin_rand_phi = o['dt'] * ((math.sin(phi) * math.sin(b_theta) * math.sin(b_phi) +
                                      math.sin(b_theta) * math.cos(b_phi) * math.cos(phi)) * math.cos(theta) -
                                      math.cos(b_theta) * math.sin(theta)) * c['gamma'] * b_mag / math.sin(theta)

        # Calculate the final position
        theta += d_theta + d_spin_rand_theta
        phi += d_phi + d_spin_rand_phi

        # Save the data to the atom
        p = self.set_position(theta, phi)

        return self.id, p

    # Integrate one step using fourth order Runge-Kutta (https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
    # This is a nonlinear method as it using partial time steps
    # This integrator does not have energy conservation for some systems
    def take_rk4_step(self, b_rand):
        # Get the current position
        o, c = self.options, self.constants
        theta, phi = self.theta, self.phi
        b_mag, b_theta, b_phi = math.fabs(b_rand[0]), b_rand[1], b_rand[2]

        # Calculate partial Runge Kutta steps
        d_theta1, d_phi1 = self.calculate_function_value(theta, phi)
        d_theta2, d_phi2 = self.calculate_function_value(theta + d_theta1 * o['dt'] / 2, phi + d_phi1 * o['dt'] / 2)
        d_theta3, d_phi3 = self.calculate_function_value(theta + d_theta2 * o['dt'] / 2, phi + d_phi2 * o['dt'] / 2)
        d_theta4, d_phi4 = self.calculate_function_value(theta + d_theta3 * o['dt'], phi + d_phi3 * o['dt'])

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
        theta += d_theta + d_spin_rand_theta * o['dt']
        phi += d_phi + d_spin_rand_phi * o['dt']

        # Save the data to the atom
        p = self.set_position(theta, phi)

        return self.id, p

    def take_euler_step(self, b_rand):
        # Calculate function value
        theta, phi = self.theta, self.phi
        d_stheta, d_sphi = self.calculate_function_value(theta, phi, b_rand)

        # Take the step and calculate the final position
        theta += d_stheta * self.options['dt']
        phi += d_sphi * self.options['dt']

        # Save the data to the atom
        p = self.set_position(theta, phi)

        # Return some info
        return self.id, p

    # Calculate energy using hamiltonian
    def get_energy(self, neighbours):
        energy = 0
        for item in self.neighbours:
            if item != self.id:
                # Hamiltonian
                energy += -2 * self.options['J'] * dot(neighbours[item].current_position(), self.current_position())

        return energy
