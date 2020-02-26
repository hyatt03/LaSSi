from pathlib import Path
import sys

p = Path(__file__).parents[1]
sys.path.append(str(p.resolve()))

import unittest
from SimulationBaseClass import BaseSimulation
from tempfile import mkdtemp
from shutil import rmtree
import numpy as np
from utils import calculate_dt_and_n, to_sph
import matplotlib.pyplot as plt


class SingleSpinTest(unittest.TestCase):
    def setUp(self):
        # Create temporary folder for data files
        self.tmpdir = mkdtemp()
        self.tmpdir = 'data/single_spins_test'

        # Initialize a simulation
        self.sim = BaseSimulation()

        # Calculate a few things
        # Start by setting a B-field with a non trivial direction
        self.larmor_B = B = [0.5, -1.3, 1.5]

        # Calculate the expected energy for Larmor precession
        self.expected_energy = - self.sim.constants['g'] * self.sim.constants['mu_b_meV'] * \
                          np.sqrt(B[2] ** 2 + B[1] ** 2 + B[0] ** 2)

        # calculate dt and N
        dt, n = calculate_dt_and_n(self.expected_energy * 0.5, self.expected_energy * 2)
        self.N = int(np.ceil(n))

        # Configure the simulation
        self.sim.options['simulation_name'] = 'TestSingleSpin'
        self.sim.options['input_file'] = 'tests/molecules/gd_ion.pdb'
        self.sim.options['transform_file'] = self.tmpdir + '/transforms.h5'
        self.sim.options['debug'] = True
        self.sim.options['spin'] = 4
        self.sim.options['J'] = 0 # No interaction, just B-field

        # Set dt using estimates from calculations.
        self.sim.options['dt'] = dt
        print(dt)

        # Load the particle
        self.sim.load_particles()

    def tearDown(self):
        self.sim.close()
        # rmtree(self.tmpdir)

    def test_temperature(self):
        B_fields = [
            [0., 0., 10.],
            [0., 0., 50.],
            [0., 0., 100.]
        ]

        # Set the damping
        self.sim.options['l'] = l = 1e-2
        self.sim.options['dt'] = 1e-14
        self.sim.options['integrator'] = 'RK4'

        for B in B_fields:
            # Reset any old results
            self.sim.close()
            self.sim.datafile = None
            self.sim.transformfile = None
            self.sim.transformtables = {}

            # Set the B-field
            self.sim.options['B'] = B

            print(B)

            # Initialize array to hold net magnetization
            magn = []

            # Get temperatures in the range 0K to 1000K, in 20 steps
            temperatures = (np.array(range(3)) + 1) * 1000 / 20
            temperatures = [5, 50, 500]

            # Iterate over the temperatures
            for T in temperatures:
                self.sim.options['T'] = T

                # Set initial condition
                r, theta, phi = to_sph([0.2, 0.2, -0.2])
                self.sim.particles.atoms[0].set_position(theta, phi)

                # Setup the filename
                self.sim.options['data_file'] = f'{self.tmpdir}/data_ad_bs_3_T{T}_B{B[2]}_l{l}.h5'

                # Inform the user of whats going on
                print(self.sim.options['data_file'])

                # Run the simulation
                self.sim.run_simulation(1e5)

                # Calculate the net magnetization
                table = self.sim.datatables['p0']
                magn.append(np.mean(table.cols.pos_z[-1000:]) / (self.sim.options['spin']))

                # plt.plot(table.cols.pos_x)
                # plt.plot(table.cols.pos_y)
                plt.plot(table.cols.pos_z)
                # plt.show()

            # Plot the magnetization as a function of temperature
            # plt.plot(temperatures, magn, '.')

        # Show the plots
        plt.show()

        self.sim.options['dt'] = self.sim.options['dt'] / 20

    def test_larmor_frequency(self):
        # Reset any old results
        self.sim.close()
        self.sim.datafile = None
        self.sim.transformfile = None
        self.sim.transformtables = {}

        # Set some weird direction for the B-field, it should still precess correctly
        self.sim.options['B'] = self.larmor_B

        # Remove temperature and damping
        self.sim.options['l'] = 0
        self.sim.options['T'] = 0

        # Set the filename
        self.sim.options['data_file'] = self.tmpdir + '/data.h5'

        # Start by running the sim
        self.sim.run_simulation(self.N)

        # Plot the components of the spin
        self.sim.plot_components_individually(str(0), 'larmor_precession.png')

        # Run a transformation
        self.sim.run_transformations(np.array([0, 0, 0]))

        # Grab the energy with largest intensity
        table = self.sim.transformtables['[0 0 0]']
        max_intensity_energy = table.cols.energy[np.argmax(table.cols.I_xx)]

        # Check that we see a peak right where we expect from Larmor precession.
        self.assertAlmostEqual(self.expected_energy, max_intensity_energy, places=3)


if __name__ == '__main__':
    # If we call the script directly, just run the test.
    unittest.main()
