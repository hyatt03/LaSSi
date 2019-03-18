import unittest
from SimulationBaseClass import BaseSimulation
from tempfile import mkdtemp
from shutil import rmtree
import numpy as np
from utils import calculate_dt_and_n


class SingleSpinTest(unittest.TestCase):
    def setUp(self):
        # Create temporary folder for data files
        self.tmpdir = mkdtemp()

        # Initialize a simulation
        self.sim = BaseSimulation()

        # Calculate a few things
        B = [0.5, -1.3, 1.5]

        # The expected energy for larmor precession
        self.expected_energy = - self.sim.constants['g'] * self.sim.constants['mu_b_meV'] * \
                          np.sqrt(B[2] ** 2 + B[1] ** 2 + B[0] ** 2)

        # dt and N
        dt, n = calculate_dt_and_n(self.expected_energy * 0.01, self.expected_energy * 2)
        dt = 1e-14
        n = 2 ** 23

        # Configure the simulation
        self.sim.options['simulation_name'] = 'TestSingleSpin'
        self.sim.options['input_file'] = 'tests/molecules/gd_ion.pdb'
        self.sim.options['data_file'] = self.tmpdir + '/data.h5'
        self.sim.options['transform_file'] = self.tmpdir + '/transforms.h5'
        self.sim.options['spin'] = 7 / 2
        self.sim.options['l'] = 0
        self.sim.options['J'] = 0
        self.sim.options['T'] = 0
        self.sim.options['debug'] = True
        self.sim.options['integrator'] = 'RK2'

        # Set some weird direction for the B-field, it should still precess correctly
        self.sim.options['B'] = B

        # Set dt using estimates from calculations.
        self.sim.options['dt'] = dt

        # Load the particle and run the sim
        self.sim.load_particles()
        self.sim.run_simulation(np.ceil(n))

    def tearDown(self):
        self.sim.close()
        rmtree(self.tmpdir)

    def test_larmor_frequency(self):
        # Initialize the simulation, run it, and get initial states
        self.sim.run_transformations(np.array([0, 0, 0]))

        max_value = max(self.sim.transformtables['[0 0 0]'].cols.I_xx)
        for row in self.sim.transformtables['[0 0 0]'].where('I_xx == {}'.format(max_value)):
            print(self.expected_energy, row['energy'])
            self.assertAlmostEqual(self.expected_energy, row['energy'], places = 3)

if __name__ == '__main__':
    unittest.main()
