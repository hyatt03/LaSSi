from pathlib import Path
import sys

p = Path(__file__).parents[1]
sys.path.append(str(p.resolve()))

import unittest
from SimulationBaseClass import BaseSimulation
import tempfile
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt
from utils import to_sph, calculate_dt_and_n


class TwoSpinTest(unittest.TestCase):
    def setUp(self):
        # Create temporary folder for data files
        self.tmpdir = tempfile.mkdtemp()
        self.tmpdir = 'data/two_spins_test'

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
        self.sim.options['J'] = - 173 * self.sim.constants['k_b']
        self.sim.options['T'] = 0
        self.sim.options['B'] = [0., 0., 0.]
        self.sim.options['integrator'] = 'ad_bs'
        self.sim.options['debug'] = True

        # Load the particles
        self.sim.load_particles()

    def test_exchange_frequence_as_deviation_angle(self):
        deviation_angles = np.array(range(1, 45, 1)) * np.pi / 180
        top_energies_z1 = []
        expected_energies = []
        mean_dot_products = []

        for idx, d_angle in enumerate(deviation_angles):
            if self.sim.options['debug'] or True:
                # Print the angle so we know how far we are
                print('Current angle', d_angle * 180 / np.pi)

            # Start by setting the initial position of the spins
            r, theta1, phi1 = to_sph([np.sin(d_angle), np.cos(d_angle), np.cos(d_angle)])
            r, theta2, phi2 = to_sph([np.sin(d_angle), -np.cos(d_angle), -np.cos(d_angle)])

            self.sim.particles.atoms[0].set_position(theta1, phi1)
            self.sim.particles.atoms[1].set_position(theta2, phi2)

            # Figure out what the effective B-field is
            self.sim.particles.combine_neighbours()
            BX = np.linalg.norm(self.sim.particles.atoms[0].B_eff)

            # Figure out what our expected energy is
            expected_energy = - self.sim.constants['Hz_to_meV'] * 2 * self.sim.constants['gamma'] * BX * np.sin(2 * d_angle)

            # dt and N
            dt, N = calculate_dt_and_n(expected_energy * 0.1, expected_energy * 10)
            N = 200001
            self.sim.options['dt'] = dt

            # Reset expected for plotting, we get som constant off
            expected_energy = - self.sim.constants['Hz_to_meV'] * 2 * self.sim.constants['gamma'] * BX * np.sin(2 * d_angle) / 13.1072056777021
            expected_energies.append(expected_energy)

            # Set a new datafile so we dont just load the data
            self.sim.options['data_file'] = self.tmpdir + f'/data_ad_bs_2_{str(d_angle)}_{str(N)}.h5'

            # Next run the simulation
            self.sim.run_simulation(np.ceil(N))

            # Grab the spin positions
            spin_1 = np.array([
                self.sim.datatables['p0'].cols.pos_x,
                self.sim.datatables['p0'].cols.pos_y,
                self.sim.datatables['p0'].cols.pos_z
            ])

            spin_2 = np.array([
                self.sim.datatables['p1'].cols.pos_x,
                self.sim.datatables['p1'].cols.pos_y,
                self.sim.datatables['p1'].cols.pos_z
            ])

            # Calculate the dot product of the spins as a function of time
            s1_d_s2 = np.array([np.dot(spin_1[:, i], spin_2[:, i]) for i in range(spin_1.shape[1])])
            mean_dot_products.append(- 2 * self.sim.options['J'] * np.mean(s1_d_s2))

            # Run a fit of the simulation
            p1, p2 = np.polyfit(range(len(s1_d_s2[3:])), s1_d_s2[3:], 1)

            # Assert conservation of energy within 10 decimal places.
            self.assertAlmostEqual(p1, 0.0, 10)

            # Run a fourier transform to get the peaks
            n = int(2 ** np.ceil(np.log2(len(spin_1[2]))))
            Z1 = np.abs(np.fft.rfft(spin_1[2], n=n))

            # Calculate the frequencies and convert that to energies
            freqs = np.fft.rfftfreq(n, self.sim.options['dt'])
            energies = freqs * self.sim.constants['Hz_to_meV']

            # Find the peak and save the energy at which it occurred
            max_idx = next(i for i,v in enumerate(energies) if v > 400)
            first_index_of_search = 2
            z1_e_max_idx = np.argmax(Z1[first_index_of_search:max_idx]) + first_index_of_search
            top_energies_z1.append(energies[z1_e_max_idx])

            # Reset the simulation for the next angle
            plt.close('all')
            self.sim.close()
            self.sim.datafile = None
            self.sim.transformfile = None
            self.sim.transformtables = {}

            if not self.sim.options['debug']:
                rmtree(self.tmpdir)

        # Plot the expected energies along with the simulated energies so we can see if there is overlap.
        plt.plot(deviation_angles * 180 / np.pi, expected_energies)
        plt.plot(deviation_angles * 180 / np.pi, top_energies_z1, '.')
        plt.show()


if __name__ == '__main__':
    unittest.main()
