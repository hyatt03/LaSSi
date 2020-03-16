from pathlib import Path
import sys

p = Path(__file__).parents[1]
sys.path.append(str(p.resolve()))

import matplotlib.pyplot as plt
import matplotlib as mpl
import unittest
from unittest import TestCase
from tempfile import mkdtemp
from shutil import rmtree
from ase import Atoms
import numpy as np

from SimulationBaseClass import BaseSimulation
from utils import to_sph, calculate_dt_and_n

k_b_meV = 8.6173303e-2


class SpinChainTest(TestCase):
    def setUp(self):
        # Create temporary folder for data files
        self.tmpdir = mkdtemp()
        self.tmpdir = 'data/ten_spins_test'

        # Initialize a simulation
        self.sim = BaseSimulation()

        # Some parameters
        n = 22  # Number of atoms
        d = 0.9  # One Aangstroem between each atom
        J = -126.0
        T = 0.0
        S = 7/2
        self.q = q = 2 * np.pi / (n * d)

        # Calculate dt and N from expected energy
        expected_energy = np.abs(8 * S * J * k_b_meV)

        dt, N = calculate_dt_and_n(expected_energy * 0.5, expected_energy * 2)
        print(expected_energy)
        self.N = N = int(N / 8)
        dt = 68 * dt
        # N = 2000

        BZ = 0.

        # Configure the simulation
        self.sim.options['simulation_name'] = 'TestSpinChain'
        self.sim.options['l'] = 0
        self.sim.options['dt'] = dt
        self.sim.options['J'] = J * self.sim.constants['k_b']
        self.sim.options['T'] = T
        self.sim.options['B'] = [0., 0., BZ]
        self.sim.options['spin'] = S
        self.sim.options['pbc'] = (True, False, False)
        self.sim.options['debug'] = True
        self.sim.options['data_file'] = self.tmpdir + f'/data_n{n}_J{J}_T{T}_N{N}_S{S}_dt{dt}_d{d}_B{BZ}.h5'
        self.sim.options['transform_file'] = self.tmpdir + f'/transforms_n{n}_J{J}_T{T}_N{N}_S{S}_dt{dt}_d{d}.h5'

        print(self.sim.options['data_file'])

        # Calculate positions of atoms
        positions = []
        for i in range(n):
            positions.append([d + i * d, 0.0001, 0.0001])

        # Create the atoms and load them into the simulation
        chain = Atoms('Gd' + str(n), positions = positions)
        self.sim.load_particles(chain)

        # Set the initial positions of the spin to mirror a spin wave state
        for q_m in range(1, 14):
            q = q_m * np.pi / 22

            realspace = []
            for i in range(n):
                x, y, z = np.cos(q * i), np.sin(q * i), 0.1

                # Perturb the system
                if i == 100 and False:
                    x += 0.1
                    y += 0.1

                r, theta, phi = to_sph([x, y, z])
                self.sim.particles.atoms[i].set_position(theta, phi)

                realspace.append([x, y, z])

            self.sim.run_simulation((q_m) * N)

    def tearDown(self):
        self.sim.close()
        if not self.sim.options['debug']:
            rmtree(self.tmpdir)

    def test_linear_spin_waves(self):
        x9 = self.sim.datatables['p9'].cols.pos_x
        x10 = x = self.sim.datatables['p10'].cols.pos_x
        x11 = self.sim.datatables['p11'].cols.pos_x
        y = self.sim.datatables['p10'].cols.pos_y
        z = self.sim.datatables['p10'].cols.pos_z

        t = np.asarray(range(len(x))) * self.sim.options['dt']
        plt.plot(t, x)
        plt.plot(t, y)
        plt.plot(t, z)

        plt.xlabel('t [s]')
        plt.ylabel('s(t) [A.U.]')

        plt.show()
        # plt.close()

        plt.plot(x9)
        plt.plot(x10)
        plt.plot(x11)
        plt.show()
        # plt.close()

        # The expected energy for larmor precession
        # expected_energy = - self.sim.constants['g'] * self.sim.constants['mu_b_meV'] * self.sim.options['B'][2]
        # max_value = max(self.sim.transformtables['[0 0 0]'].cols.I_xx)
        # for row in self.sim.transformtables['[0 0 0]'].where('I_xx == {}'.format(max_value)):
        #     self.assertAlmostEqual(expected_energy, row['energy'], places=4)

if __name__ == '__main__':
    unittest.main()
