import unittest
from SimulationBaseClass import BaseSimulation
from tempfile import mkdtemp
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt


class ThreeSpinTest(unittest.TestCase):
    def setUp(self):
        # Create temporary folder for data files
        self.tmpdir = mkdtemp()

        # Initialize a simulation
        self.sim = BaseSimulation()

        # Configure the simulation
        self.sim.options['simulation_name'] = 'TestThreeSpins'
        self.sim.options['input_file'] = 'tests/molecules/three_spins_triangle.pdb'
        self.sim.options['data_file'] = self.tmpdir + '/data.h5'
        self.sim.options['transform_file'] = self.tmpdir + '/transforms.h5'
        self.sim.options['spin'] = 7 / 2
        self.sim.options['l'] = 0
        self.sim.options['dt'] = 1e-14
        self.sim.options['J'] = - 5 * self.sim.constants['k_b']
        self.sim.options['T'] = 0
        self.sim.options['B'] = [0., 0., 0.]
        self.sim.options['debug'] = True

        self.sim.load_particles()
        # self.sim.run_anneal(2 ** 14)
        self.sim.run_simulation(2 ** 20)

    def tearDown(self):
        self.sim.close()
        rmtree(self.tmpdir)

    def test_exchange_through_angles(self):
        self.sim.plot_spins_xyz('three_spins_xyz.png')
        plt.close('all')
        # plt.figure(0)
        # self.sim.plot_single_spin_xyz(0, 'first_spin.png')
        # plt.figure(1)
        # self.sim.plot_single_spin_xyz(1, 'second_spin.png')
        # plt.figure(2)
        # self.sim.plot_single_spin_xyz(2, 'third_spin.png')

        self.sim.animate_spins_xyz('three_spin_animation.gif', 12, 1200)


    # def test_larmor_frequency(self):
    #     # Initialize the simulation, run it, and get initial states
    #     self.sim.run_transformations(np.array([0, 0, 0]))
    #
    #     # The expected energy for larmor precession
    #     expected_energy = - self.sim.constants['g'] * self.sim.constants['mu_b_meV'] * self.sim.options['B'][2]
    #     max_value = max(self.sim.transformtables['[0 0 0]'].cols.I_xx)
    #     for row in self.sim.transformtables['[0 0 0]'].where('I_xx == {}'.format(max_value)):
    #         self.assertAlmostEqual(expected_energy, row['energy'], places=4)

if __name__ == '__main__':
    unittest.main()
