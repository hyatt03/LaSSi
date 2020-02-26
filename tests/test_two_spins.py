"""
WARNING: This is a very slow test
"""

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
        self.sim.options['J'] = - 173 * self.sim.constants['k_b']
        self.sim.options['T'] = 0
        self.sim.options['B'] = [0., 0., 0.]
        self.sim.options['debug'] = True

        # Load the particles
        self.sim.load_particles()

    def test_dampening(self):
        # Set initial conditions
        d_angle_degrees = 45
        d_angle = d_angle_degrees * np.pi / 180

        r, theta1, phi1 = np.round(to_sph([np.sin(d_angle), np.cos(d_angle), np.cos(d_angle)]), 3)
        r, theta2, phi2 = np.round(to_sph([np.sin(d_angle), -np.cos(d_angle), -np.cos(d_angle)]), 3)

        self.sim.particles.atoms[0].set_position(theta1, phi1)
        self.sim.particles.atoms[1].set_position(theta2, phi2)

        # Figure out what the effective B-field is
        self.sim.particles.combine_neighbours()
        B_eff = self.sim.particles.atoms[0].B_eff

        # Figure out what our expected energy is
        expected_energy = (
                (self.sim.constants['Hz_to_meV'] / (2 * np.pi)) *
                self.sim.constants['gamma'] *
                (B_eff[0] * np.cos(d_angle) - B_eff[2] * np.sin(d_angle)) / np.cos(d_angle))

        # calculate and set dt and N
        dt, N = calculate_dt_and_n(expected_energy * 0.5, expected_energy * 2)
        N = 4 * int(np.ceil(N))
        self.sim.options['dt'] = 2 * dt

        # Next run the simulation
        self.sim.options['data_file'] = self.tmpdir + '/data_without_dampening.h5'
        self.sim.run_simulation(np.ceil(N))

        # Setup the baseline comparison
        x = self.sim.datatables['p0'].cols.pos_x
        y = self.sim.datatables['p0'].cols.pos_y
        z = self.sim.datatables['p0'].cols.pos_z

        t = np.asarray(range(len(x))) * self.sim.options['dt']

        fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
        ax = axs.flat[0]

        ax.plot(t, x, label='x')
        ax.plot(t, y, label='y')
        ax.plot(t, z, label='z')

        ax.set_title('a', fontsize=10)
        ax.set_xlabel('t [s]', fontsize=10)
        ax.set_ylabel('s(t) [A.U.]', fontsize=10)
        ax.legend(prop={'size': 10})

        ax = axs.flat[1]

        ax.plot(t[-2000:], x[-2000:], label='x')
        ax.plot(t[-2000:], y[-2000:], label='y')
        ax.plot(t[-2000:], z[-2000:], label='z')

        ax.set_title('b', fontsize=10)
        ax.set_xlabel('t [s]', fontsize=10)
        ax.set_ylabel('s(t) [A.U.]', fontsize=10)
        ax.legend(prop={'size': 10})

        # plt.figure(figsize=(6, 4), tight_layout=True)
        #
        # plt.plot(t, x, label='x')
        # plt.plot(t, y, label='y')
        # plt.plot(t, z, label='z')
        #
        # plt.xlabel('t [s]', fontsize=10)
        # plt.ylabel('s(t) [A.U.]', fontsize=10)
        # plt.legend(prop={'size': 10})
        #
        # plt.yticks(fontsize=9)
        # plt.xticks(fontsize=9)

        # plt.show()

        # Reset the simulation so we can compare with dampening enabled
        self.sim.close()
        self.sim.datafile = None
        self.sim.transformfile = None
        self.sim.transformtables = {}

        # Set dampening
        self.sim.options['l'] = 1e-3

        # Run the second simulation
        self.sim.options['data_file'] = self.tmpdir + '/data_with_dampening.h5'
        self.sim.run_simulation(np.ceil(N))

        # And plot again so we can compare
        x = self.sim.datatables['p0'].cols.pos_x
        y = self.sim.datatables['p0'].cols.pos_y
        z = self.sim.datatables['p0'].cols.pos_z

        t = np.asarray(range(len(x))) * self.sim.options['dt']

        # plt.figure(figsize=(6, 4), tight_layout=True)

        ax = axs.flat[2]

        ax.plot(t, x, label='x')
        ax.plot(t, y, label='y')
        ax.plot(t, z, label='z')

        ax.set_title('c', fontsize=10)
        ax.set_xlabel('t [s]', fontsize=10)
        ax.set_ylabel('s(t) [A.U.]', fontsize=10)
        ax.legend(prop={'size': 10})

        ax = axs.flat[3]

        ax.plot(t[-2000:], x[-2000:], label='x')
        ax.plot(t[-2000:], y[-2000:], label='y')
        ax.plot(t[-2000:], z[-2000:], label='z')

        ax.set_title('d', fontsize=10)
        ax.set_xlabel('t [s]', fontsize=10)
        ax.set_ylabel('s(t) [A.U.]', fontsize=10)
        ax.legend(prop={'size': 10})

        plt.show()

    def test_exchange_frequence_as_deviation_angle(self):
        deviation_angles = np.array(range(1, 89, 1)) * 1.0 # degree steps
        top_energies_z1 = []
        expected_energies = []
        ratios = []

        for idx, d_angle_degrees in enumerate(deviation_angles):
            # Convert to radians
            d_angle = d_angle_degrees * np.pi / 180

            # Start by setting the initial position of the spins
            r, theta1, phi1 = np.round(to_sph([np.sin(d_angle), np.cos(d_angle), np.cos(d_angle)]), 3)
            r, theta2, phi2 = np.round(to_sph([np.sin(d_angle), -np.cos(d_angle), -np.cos(d_angle)]), 3)

            self.sim.particles.atoms[0].set_position(theta1, phi1)
            self.sim.particles.atoms[1].set_position(theta2, phi2)

            # Figure out what the effective B-field is
            self.sim.particles.combine_neighbours()
            B_eff = self.sim.particles.atoms[0].B_eff

            # Figure out what our expected energy is
            expected_energy = (
                    (self.sim.constants['Hz_to_meV'] / (2 * np.pi)) *
                    self.sim.constants['gamma'] *
                    (B_eff[0] * np.cos(d_angle) - B_eff[2] * np.sin(d_angle)) / np.cos(d_angle))
            expected_energies.append(expected_energy)

            # dt and N
            dt, N = calculate_dt_and_n(expected_energy * 0.1, expected_energy * 10)
            N = int(np.ceil(N))
            self.sim.options['dt'] = dt

            # Set a new datafile so we dont just load the data
            filename = f'{self.tmpdir}/data_{self.sim.options["integrator"]}_angle_{str(d_angle_degrees)}_N_{str(N)}.h5'
            self.sim.options['data_file'] = filename

            if self.sim.options['debug']:
                # Print the filename as it contains the relevant info.
                print('Current filename', filename)

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

            # Run a fit of the simulation
            p1, p2 = np.polyfit(range(len(s1_d_s2[3:])), s1_d_s2[3:], 1)

            # Assert conservation of energy within 10 decimal places.
            self.assertAlmostEqual(p1, 0.0, 10)

            # Run a fourier transform to get the peaks
            n = int(2 ** np.ceil(np.log2(len(spin_1[2]))))
            Z1 = np.abs(np.fft.fft(spin_2[2], n=n))

            # Calculate the frequencies and convert that to energies
            freqs = np.fft.fftfreq(n, self.sim.options['dt'])
            energies = freqs * self.sim.constants['Hz_to_meV']

            # Find the peak and save the energy at which it occurred
            try:
                max_idx = next(i for i,v in enumerate(energies) if v > 400)
            except:
                max_idx = -1
            first_index_of_search = 2
            z1_e_max_idx = np.argmax(Z1[first_index_of_search:max_idx]) + first_index_of_search
            top_energies_z1.append(energies[z1_e_max_idx])

            # Check the ratio of expected to actual
            ratios.append(expected_energy / top_energies_z1[-1])

            # Reset the simulation for the next angle
            self.sim.close()
            self.sim.datafile = None
            self.sim.transformfile = None
            self.sim.transformtables = {}

            # If were not just debugging we want to delete the datafile
            if not self.sim.options['debug']:
                rmtree(self.tmpdir)

        # Plot the expected along with the actual energies
        if self.sim.options['debug']:
            plt.figure(figsize=(6, 4), tight_layout=True)
            plt.plot(deviation_angles, expected_energies, label='Expected')
            plt.plot(deviation_angles, top_energies_z1, '.', label='Numerical')
            plt.xlabel('Angle [Degrees]', fontsize=10)
            plt.ylabel('Resonance energy [meV]', fontsize=10)
            plt.yticks(fontsize=9)
            plt.xticks(fontsize=9)
            plt.legend(prop={'size': 10})
            plt.savefig('resonance_of_angle_two_spins.png', dpi=300)
            plt.show()

        # Assert that the ratio between the expected energy and the actual energy is around 1
        # Thereby validating the exchange.
        for ratio in ratios:
            self.assertAlmostEqual(ratio, 1, 1)


if __name__ == '__main__':
    unittest.main()
