from pathlib import Path
import sys

p = Path(__file__).parents[1]
sys.path.append(str(p.resolve()))

import unittest
from unittest import TestCase
from SimulationBaseClass import BaseSimulation
from tempfile import mkdtemp
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt

from ase import Atoms

class SpinChainTest(TestCase):
    def setUp(self):
        # Create temporary folder for data files
        self.tmpdir = mkdtemp()
        self.tmpdir = 'data/ten_spins_test'

        # Initialize a simulation
        self.sim = BaseSimulation()

        # Configure the simulation
        self.sim.options['simulation_name'] = 'TestSpinChain'
        self.sim.options['data_file'] = self.tmpdir + '/data.h5'
        self.sim.options['transform_file'] = self.tmpdir + '/transforms.h5'
        self.sim.options['l'] = 0
        self.sim.options['dt'] = 1e-14
        self.sim.options['J'] = 65.8 * self.sim.constants['k_b']
        self.sim.options['T'] = 0.000
        self.sim.options['B'] = [0., 0., 5]
        self.sim.options['spin'] = 5/2
        self.sim.options['pbc'] = (True, False, False)
        self.sim.options['debug'] = True

        positions = []
        n = 100 # Number of atoms
        d = 1 # One Aangstroem between each atom
        for i in range(n):
            positions.append([d, d, i * d + d])

        chain = Atoms('Gd' + str(n), positions = positions)

        self.sim.load_particles(chain)
        self.sim.run_anneal(int(2**8))
        self.sim.run_simulation(int(2**16))

    def tearDown(self):
        self.sim.close()
        # rmtree(self.tmpdir)

    def test_linear_spin_waves(self):
        I_of_omega = np.zeros((32768, 201))
        hbar_omega = np.zeros((32768, 1))
        qs = []
        for q_m in range(201):
            # Determine scattering vector and run transformations for that scattering vector
            q = np.array([round((-0.5 + 1/200 * q_m) * (6.28 / 1.1), 3), 0., 0.])
            qs.append(q[0])
            transformation_table = self.sim.run_transformations(q)

            # Grab the results
            for idx, row in enumerate(transformation_table):
                I_of_omega[idx, q_m] = np.log(row['I_xx'])
                hbar_omega[idx, 0] = (row['frequency'] * self.sim.constants['Hz_to_meV'])

        print(I_of_omega)
        print(hbar_omega)

        print(I_of_omega.shape)

        max_freq_idx = 100
        extent = [qs[0], qs[-1], hbar_omega[1, 0], hbar_omega[max_freq_idx, 0]]
        im = plt.imshow(I_of_omega[1:max_freq_idx, :], extent=extent)
        ax = plt.gca()
        ax.set_aspect(abs(extent[1] - extent[0]) / abs(extent[3] - extent[2]))
        cbar = ax.figure.colorbar(im, ax=ax)
        plt.show()

        # And plot them
        # plt.plot(hbar_omega, I_of_omega)
        # plt.ylim(0, 10)
        # plt.show()

        # Run the simulation as a function of temperature
        # run_T = False
        # for T in np.arange(0.05, 150.05, 0.05):
        #     if run_T:
        #         self.sim.run_simulation(int(2), keep_going=True)
        #
        # # Initialize the simulation, run it, and get initial states
        # qs = np.arange(0, 0.5, 0.01)
        # I_of_q = []
        # for q_size in qs:
        #     avg_I = 0
        #
        #     for q_dir in [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]:
        #         q = q_size * q_dir
        #         transformation_table = self.sim.run_transformations(q)
        #         print(transformation_table)
        #         # avg_I += (1/9) * (row['I_xx'] + row['I_yy'] + row['I_zz'])
        #
        #     I_of_q.append(avg_I)
        #
        # plt.plot(qs, I_of_q)
        # plt.show()

        # self.sim.plot_qxx_vs_qyy(self.tmpdir + '/qxx_qyy.png')

        # self.sim.plot_spins_xy(self.tmpdir + '/spin_xy.png')
        # self.sim.plot_system_energies_as_f_of_t(self.tmpdir + '/total_e.png')

        # self.sim.plot_spins_xyz(self.tmpdir + '/spin_xyz.png')
        # plt.show()

        # self.sim.animate_spins_xyz(self.tmpdir + '/spin_xyz.gif', 2, 6000)

        # self.sim.plot_positions_xy(self.tmpdir + '/position_xy.png')
        # self.sim.plot_scattering_cross_section(self.tmpdir + '/cross_section_{}.png')

        # The expected energy for larmor precession
        # expected_energy = - self.sim.constants['g'] * self.sim.constants['mu_b_meV'] * self.sim.options['B'][2]
        # max_value = max(self.sim.transformtables['[0 0 0]'].cols.I_xx)
        # for row in self.sim.transformtables['[0 0 0]'].where('I_xx == {}'.format(max_value)):
        #     self.assertAlmostEqual(expected_energy, row['energy'], places=4)

if __name__ == '__main__':
    unittest.main()
