from pathlib import Path
import sys

p = Path(__file__).parents[1]
sys.path.append(str(p.resolve()))

import unittest
from SimulationBaseClass import BaseSimulation
from tempfile import mkdtemp
from shutil import rmtree
import random
import numpy as np


class BasicFunctionalityTest(unittest.TestCase):
    def setUp(self):
        # Create temporary folder for data files
        self.tmpdir = mkdtemp()

        # Initialize a simulation
        sim = BaseSimulation()

        # Configure the simulation
        sim.options['simulation_name'] = 'TestBasic'
        sim.options['input_file'] = 'tests/molecules/gd_ion.pdb'
        sim.options['data_file'] = self.tmpdir + '/data.h5'
        sim.options['transform_file'] = self.tmpdir + '/transforms.h5'
        sim.options['spin'] = 7 / 2
        sim.options['l'] = 5e-4
        sim.options['dt'] = 1e-10
        sim.options['J'] = 1e-6
        sim.options['T'] = 0.05
        sim.options['B'] = [0., 0., 0.1]

        self.sim = sim

        # Seed the random number generator (we don't actually want random during testing)
        random.seed(0)

    def tearDown(self):
        self.sim.close()
        rmtree(self.tmpdir)

    # Tests if the module can load particles at all
    def test_load_particles(self):
        self.assertIsNone(self.sim.particles)
        self.sim.load_particles()
        self.assertIsNotNone(self.sim.particles)

    # Tests if anything happens when we anneal
    def test_anneal(self):
        # Load the particles and find the initial position
        self.sim.load_particles()
        initial_position = self.sim.particles.atoms[0].pos.copy()

        # Run the anneal and find the final position
        self.sim.run_anneal(30)
        final_position = self.sim.particles.atoms[0].pos

        # Check something happened
        self.assertNotAlmostEqual(initial_position[0], final_position[0], places=4)
        self.assertNotAlmostEqual(initial_position[1], final_position[1], places=4)
        self.assertNotAlmostEqual(initial_position[2], final_position[2], places=4)

    def test_simulation(self):
        # Initialize the simulation and get initial states
        self.sim.load_particles()
        N_iterations = 2**4
        initial_position = self.sim.particles.atoms[0].pos.copy()
        self.assertDictEqual(self.sim.datatables, {})

        # Run the simulation and save final results
        self.sim.run_simulation(N_iterations)
        final_position = self.sim.particles.atoms[0].pos
        firstkey = list(self.sim.datatables.keys())[0]

        # Check something happened to positions
        self.assertNotAlmostEqual(initial_position[0], final_position[0], places=4)
        self.assertNotAlmostEqual(initial_position[1], final_position[1], places=4)
        self.assertNotAlmostEqual(initial_position[2], final_position[2], places=4)

        # Check the data was saved
        self.assertEqual(firstkey, 'p0')
        self.assertEqual(len(self.sim.datatables[firstkey]), N_iterations)

    def test_transformations(self):
        # Initialize the simulation, run it, and get initial states
        self.sim.load_particles()
        self.sim.run_simulation(2 ** 4)
        self.assertDictEqual(self.sim.transformtables, {})

        # Run the transformation
        self.sim.run_transformations(np.array([1., 0, 0]))
        self.assertEqual(len(list(self.sim.transformtables.keys())), 1)

        # Run a second transformation
        self.sim.run_transformations(np.array([1.1, 0, 0]))
        self.assertEqual(len(list(self.sim.transformtables.keys())), 2)

        # Repeat a transformation
        self.sim.run_transformations(np.array([1., 0, 0]))
        self.assertEqual(len(list(self.sim.transformtables.keys())), 2)

    def test_multiple_atoms(self):
        # Basic setup
        self.sim.options['input_file'] = 'tests/molecules/ten_spins_chain.pdb'

        # Ensure were not cheating our selves
        self.assertIsNone(self.sim.particles)

        # Load the actual particles
        self.sim.load_particles()

        # Check that they loaded
        self.assertEqual(len(self.sim.particles.atoms), 10)

        # Check that the ends have one neighbour
        self.assertEqual(len(self.sim.particles.atoms[0].neighbours), 1)
        self.assertEqual(len(self.sim.particles.atoms[9].neighbours), 1)

        # Check that one of the middle ones have 2 neighbours
        self.assertEqual(len(self.sim.particles.atoms[1].neighbours), 2)

    def test_periodic_boundary_conditions(self):
        # Basic setup
        self.sim.options['input_file'] = 'tests/molecules/ten_spins_chain.pdb'
        self.sim.options['pbc'] = (True, False, False)
        self.sim.load_particles()

        # Check that they loaded
        self.assertEqual(len(self.sim.particles.atoms), 10)

        # Check that all the atoms have two neighbours
        for a in self.sim.particles.atoms:
            self.assertEqual(len(a.neighbours), 2)


if __name__ == '__main__':
    unittest.main()
