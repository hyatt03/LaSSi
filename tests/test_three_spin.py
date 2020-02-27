from pathlib import Path
import sys

p = Path(__file__).parents[1]
sys.path.append(str(p.resolve()))

import unittest
from SimulationBaseClass import BaseSimulation
from tempfile import mkdtemp
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt
from utils import to_sph, calculate_dt_and_n


class ThreeSpinTest(unittest.TestCase):
    def setUp(self):
        # Create temporary folder for data files
        self.tmpdir = mkdtemp()
        self.tmpdir = 'data/three_spins_test'

        dt, n = calculate_dt_and_n(0.01, 10)
        n = int(n)
        J = - 172

        print(dt, n)

        # Initialize a simulation
        self.sim = BaseSimulation()
        # Configure the simulation
        self.sim.options['simulation_name'] = 'TestThreeSpins'
        self.sim.options['input_file'] = 'tests/molecules/three_spins_triangle.pdb'
        self.sim.options['spin'] = 7 / 2
        self.sim.options['l'] = l = 5e-4
        self.sim.options['dt'] = dt = 1e-14
        self.sim.options['J'] = J * self.sim.constants['k_b']
        self.sim.options['T'] = T = .05
        self.sim.options['B'] = [0., 0., 0.]
        self.sim.options['debug'] = True
        self.sim.options['data_file'] = self.tmpdir + f'/data_ad_bs_ks5_B0.1_dt{dt}_T{T}_l{l}_J{J}_60_deg.h5'
        self.sim.options['data_file'] = self.tmpdir + f'/data_dt4.1356673300000004e-16_T0.05_l5e-05_n999999.95_J-710.h5'
        self.sim.options['data_file'] = self.tmpdir + f'/data_dt1e-15_T0.05.h5'
        self.sim.options['transform_file'] = self.tmpdir + f'/transforms_final2.h5'
        self.sim.options['integrator'] = 'ad_bs'

        self.sim.load_particles()

        # Start by setting the initial position of the spins
        d_angle = 12 * np.pi / 180
        r, theta1, phi1 = to_sph([np.sin(d_angle), np.cos(d_angle), np.cos(d_angle)])
        r, theta2, phi2 = to_sph([np.sin(-d_angle), np.cos(d_angle), np.cos(d_angle)])
        r, theta3, phi3 = to_sph([np.sin(d_angle), np.cos(-d_angle), np.cos(-d_angle)])
        self.sim.particles.atoms[0].set_position(theta1, phi1)
        self.sim.particles.atoms[1].set_position(theta2, phi2)
        self.sim.particles.atoms[2].set_position(theta3, phi3)

        print([-np.sin(d_angle), np.cos(d_angle), np.cos(d_angle)])
        print([np.sin(-d_angle), -np.cos(d_angle), -np.cos(d_angle)])
        print([np.sin(d_angle), np.cos(-d_angle), np.cos(-d_angle)])

        # Get the data loaded
        self.sim.run_simulation(1)

        # restore to last position
        for p in range(0, 3):
            r, theta, phi = to_sph([
                self.sim.datatables[f'p{p}'].cols.pos_x[-1],
                self.sim.datatables[f'p{p}'].cols.pos_y[-1],
                self.sim.datatables[f'p{p}'].cols.pos_z[-1]
            ])
            self.sim.particles.atoms[p].set_position(theta, phi)

        # self.sim.run_anneal(2**18)
        self.sim.run_simulation(2**22)


    def tearDown(self):
        self.sim.close()
        # rmtree(self.tmpdir)

    def test_exchange_through_angles(self):
        x = self.sim.datatables['p0'].cols.pos_x
        y = self.sim.datatables['p0'].cols.pos_y
        z = self.sim.datatables['p0'].cols.pos_z

        # x = x[:int(len(x)/4)]
        # y = y[:int(len(y)/4)]
        # z = z[:int(len(z)/4)]

        t = np.asarray(range(len(x))) * self.sim.options['dt']

        fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
        ax = axs.flat[0]

        ax.plot(t, x, label='x')
        ax.plot(t, y, label='y')
        ax.plot(t, z, label='z')

        ax.set_title('a', fontsize=10)
        ax.set_xlabel('t [s]', fontsize=10)
        ax.set_ylabel('s(t) [A.U.]', fontsize=10)
        ax.legend(prop={'size': 10})

        ax = axs.flat[1]

        ax.plot(t, x, label='x')
        ax.plot(t, y, label='y')
        ax.plot(t, z, label='z')

        ax.set_title('b', fontsize=10)
        ax.set_xlabel('t [s]', fontsize=10)
        ax.set_ylabel('s(t) [A.U.]', fontsize=10)
        ax.legend(prop={'size': 10})

        plt.show()

        q_vectors = []
        number_of_steps = 10
        max_q = 3.6
        for q_m in range(0, number_of_steps):
            q_vectors.append(np.array([q_m * max_q / number_of_steps, 0, 0]))

        self.sim.plot_cross_section(q_vectors, 1, 'x')
        plt.show()

if __name__ == '__main__':
    unittest.main()
