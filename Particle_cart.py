# -*- coding: utf-8 -*-
"""
Class Particle wraps a OBAtom with a few extra goodies. Cleaner than extending swig++ classes.
The version with cartesian coordinates
"""

import numpy as np
import math
from utils import cross, dot

# Adams-Bashforth integration constants
ab = [1901 / 720, 2774 / 720, 2616 / 720, 1274 / 720, 251 / 720] 

class Particle_cart(object):
    def __init__(self, id, atom, N, neighbours, options, constants):
        self.id = id
        self.N = N
        self.B_eff = np.array([0.0, 0.0, 0.0], dtype='float')
        self.options = options
        self.constants = constants
        self.neighbours = neighbours
        self.lattice_position = atom.position

        self.pos = self.lattice_position / math.sqrt(dot(self.lattice_position, self.lattice_position))
        
        # We calculate a constant once, so we don't have to do it for each iteration
        self.b_eff_p = -2 * options['J'] * options['spin'] / (constants['g'] * constants['mu_b'])
        
        # We set the previous steps for use with Adams Bashforth integration
        self.spos4, self.spos3, self.spos2, self.spos1 = None, None, None, None 
        

    # Needs to be spin position, not lattice position, J will take care of interactions
    def current_position(self):
        return self.pos.copy()
    
    # Sets the position in memory
    def set_position(self, p):
        self.pos = p

    # Calculate the effective B field for this atom from its nearest neighbours
    def combine_neighbours(self, neighbours):
        o = self.options
        # We start with a copy of the external field
        self.B_eff = np.copy(o['B']) 
        
        if o['anisotropy'] is not None:
            # Grab the anisotropic field for this specific site
            B_anisotropic = np.copy(o['anisotropy'][self.id, :, :]) 

            # Multiply anisotropy field with spin position
            B_anis = np.dot(B_anisotropic, self.current_position()) 
            self.B_eff += -B_anis
        
        # Iterate over the neighbours to find the effective B field
        for item in self.neighbours:
            if item != self.id:
                self.B_eff += self.b_eff_p * neighbours[item].current_position()
                
    # Actually calculate the step.
    def calculate_function_value(self, p):
        o, c, B = self.options, self.constants, self.B_eff
        
        return c['gamma'] * (cross(p, B) + o['l']*(B * dot(p, p) - p * dot(p, B))) 

    # Evaluate the fifth step using function values from the past 4 steps
    def fifth_ad_bs_step(self, p):
        d_fp = self.calculate_function_value(p)

        # New weighted d_spos
        dp = ab[0] * d_fp - ab[1] * self.spos4 + ab[2] * self.spos3 - ab[3] * self.spos2 + ab[4] * self.spos1

        # Return d_stheta, d_sphi as well as the function values
        return dp, d_fp

    # Evaluate the fourth step using function values from the past 3 steps
    def fourth_ad_bs_step(self, p):
        d_fpos = self.calculate_function_value(p)
        return (55 / 24) * d_fpos - (59 / 24) * self.spos4 + (37 / 24) * self.spos3 - (9 / 24) * self.spos2, d_fpos

    # Evaluate the third step using function values from the past 2 steps
    def third_ad_bs_step(self, p):
        d_fpos = self.calculate_function_value(p)
        return (23 / 12) * d_fpos - (16 / 12) * self.spos4 + (5 / 12) * self.spos3, d_fpos

    # Evaluate the second step using function values from the past step
    def second_ad_bs_step(self, p):
        d_fpos = self.calculate_function_value(p)
        return (3 / 2) * d_fpos - (1 / 2) * self.spos4, d_fpos

    # Evaluate the first step (basically forward Euler)
    def first_ad_bs_step(self, p):
        return self.calculate_function_value(p)

    def take_ad_bs_step(self, b_rand):
        # Grab the constants
        o, c = self.options, self.constants
        pos = self.current_position()

        # Gradually increase steps in the Adams Bashforth method
        if self.spos4 is None:
            d_spos = self.first_ad_bs_step(pos)
            d_fpos = d_spos
        elif self.spos3 is None:
            d_spos, d_fpos = self.second_ad_bs_step(pos)
        elif self.spos2 is None:
            d_spos, d_fpos = self.third_ad_bs_step(pos)
        elif self.spos1 is None:
            d_spos, d_fpos = self.fourth_ad_bs_step(pos)
        else:
            d_spos, d_fpos = self.fifth_ad_bs_step(pos)

        # Move the values along so we keep continuing
        self.spos4, self.spos3, self.spos2, self.spos1 = d_fpos, self.spos4, self.spos3, self.spos2
        
        # Calculate the random energy added
        # This is based on temperature
        d_spin_rand = c['gamma'] * cross(pos, b_rand)
        
        # Take the step and calculate the final position and normalise the vector
        pos = pos + d_spos * o['dt'] + d_spin_rand
        pos = pos / math.sqrt(dot(pos, pos))
        
        # Save the data to the atom
        self.set_position(pos)

        return self.id, pos

    def take_RK4_step(self, b_rand):
        o, c = self.options, self.constants
        # Get the current position
        p = self.current_position()

        # Calculate partial Runge Kutta steps
        k1 = self.calculate_function_value(p)
        k2 = self.calculate_function_value(p + k1 * o['dt'] / 2) # Needs B_eff with spins p+k1
        k3 = self.calculate_function_value(p + k2 * o['dt'] / 2)
        k4 = self.calculate_function_value(p + k3 * o['dt'])

        # Calculate the total difference in the spin
        d_spin = (k1 + 2 * k2 + 2 * k3 + k4) * o['dt'] / 6

        # Calculate the random energy added
        # This is based on temperature
        d_spin_rand = c['gamma'] * cross(p, b_rand)

        # Calculate new position and normalise the vector
        p += d_spin + d_spin_rand
        p = p / math.sqrt(dot(p, p))

        # Save the data to the atom
        self.set_position(p)

        return self.id, p

    def take_RK2_step(self, b_rand):
        o, c = self.options, self.constants
        # Get the current position
        p = self.current_position()

        # Calculate partial Runge Kutta steps
        k1 = self.calculate_function_value(p)
        self.set_position( p + k1*o['dt'] )
        # self.combine_neighbours # We need to call this, but then we have to call from particles 
        
        k2 = self.calculate_function_value(self.pos )

        # Calculate the total difference in the spin
        # d_spin = (k1 + k2)/2*o['dt']
        d_spin = k2*o['dt']

        # Calculate the random energy added
        # This is based on temperature
        d_spin_rand = c['gamma'] * cross(p, b_rand)

        # Calculate new position and normalise the vector
        p += d_spin + d_spin_rand
        p = p / math.sqrt(dot(p, p))

        # Save the data to the atom
        self.set_position(p)

        return self.id, p      

    # Calculate energy using hamiltonian
    def get_energy(self, neighbours):
        energy = 0
        for item in self.neighbours:
            if item != self.id:
                # Hamiltonian
                energy += -2 * self.options['J'] * dot(neighbours[item].current_position(), self.current_position())

        return energy