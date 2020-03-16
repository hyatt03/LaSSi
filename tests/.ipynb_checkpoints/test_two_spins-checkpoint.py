#!/usr/bin/env python3

import unittest
from SimulationBaseClass import BaseSimulation
import tempfile
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt
from utils import to_sph


class TwoSpinTest(unittest.TestCase):
    def setUp(self):
        # Create temporary folder for data files
        self.tmpdir = tempfile.mkdtemp()

        # Initialize a simulation
        self.sim = BaseSimulation()

        # Configure the simulation
        self.sim.options['simulation_name'] = 'TestTwoSpins'
        self.sim.options['input_file'] = 'tests/molecules/two_spins.pdb'
        self.sim.options['data_file'] = self.tmpdir + '/data.h5'
        self.sim.options['transform_file'] = self.tmpdir + '/transforms.h5'
        self.sim.options['spin'] = 7 / 2
        self.sim.options['l'] = 0
        self.sim.options['dt'] = 1e-18
        self.sim.options['J'] = 173 * self.sim.constants['k_b']
        self.sim.options['T'] = 0
        self.sim.options['B'] = [0., 0., 0.]
        self.sim.options['debug'] = True

        self.sim.load_particles()

    def tearDown(self):
        self.sim.close()
        rmtree(self.tmpdir)

    def test_exchange_frequence_as_deviation_angle(self):
        deviation_angles = np.array(list(range(30, 45, 5))) * np.pi / 180

        for idx, d_angle in enumerate(deviation_angles):
            # Start by setting the initial position of the spins
            r, theta1, phi1 = to_sph([np.cos(d_angle), np.sin(d_angle), 0])
            r, theta2, phi2 = to_sph([-np.cos(d_angle), np.sin(d_angle), 0])

            self.sim.particles.atoms[0].set_position(theta1, phi1)
            self.sim.particles.atoms[1].set_position(theta2, phi2)

            # Next run the simulation
            self.sim.run_simulation(2 ** 19)

            # Run the transformation
            self.sim.run_transformations(np.array([0, 0, 0]))

            # max_value_xx = max(self.sim.transformtables['[0 0 0]'].cols.I_xx)
            # max_value_yy = max(self.sim.transformtables['[0 0 0]'].cols.I_yy)
            # max_value_zz = max(self.sim.transformtables['[0 0 0]'].cols.I_zz)
            #
            # for row in self.sim.transformtables['[0 0 0]'].where('I_xx == {}'.format(max_value_xx)):
            #     print('I_xx', max_value_xx, ', d_theta =', d_angle, ', E = ', row['energy'])
            #
            # for row in self.sim.transformtables['[0 0 0]'].where('I_yy == {}'.format(max_value_yy)):
            #     print('I_yy', max_value_yy, ', d_theta =', d_angle, ', E = ', row['energy'])
            #
            # for row in self.sim.transformtables['[0 0 0]'].where('I_zz == {}'.format(max_value_zz)):
            #     print('I_zz', max_value_yy, ', d_theta =', d_angle, ', E = ', row['energy'])

            self.sim.plot_energies('data/two_spins_test/t_two_spin_E_{}_' + str(d_angle) + '.png')
            plt.show()
            # self.sim.plot_spins_xyz('data/two_spins_test/two_spins_xyz_{}.png'.format(d_angle))

            # Reset the simulation for the next angle
            plt.close('all')
            self.sim.close()
            self.sim.datafile = None
            self.sim.transformfile = None
            self.sim.transformtables = {}
            rmtree(self.tmpdir)

        # Initialize the simulation, run it, and get initial states
        # self.sim.run_transformations(np.array([0, 0, 0]))
        # self.sim.plot_energies('two_spins_energy_{}.png')

        # max_value = max(self.sim.transformtables['[0 0 0]'].cols.I_xx)

        # The expected energy for larmor precession
        # expected_energy = - self.sim.constants['g'] * self.sim.constants['mu_b_meV'] * self.sim.options['B'][2]
        # max_value = max(self.sim.transformtables['[0 0 0]'].cols.I_xx)
        # for row in self.sim.transformtables['[0 0 0]'].where('I_xx == {}'.format(max_value)):
            # self.assertAlmostEqual(expected_energy, row['energy'], places=4)

if __name__ == '__main__':
    unittest.main()
