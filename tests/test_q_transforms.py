from pathlib import Path
import sys

p = Path(__file__).parents[1]
sys.path.append(str(p.resolve()))

import matplotlib.pyplot as plt
import unittest
from unittest import TestCase
from tempfile import mkdtemp
from shutil import rmtree
from ase import Atoms
import numpy as np

from SimulationBaseClass import BaseSimulation
from utils import to_sph, calculate_dt_and_n

k_b_meV = 8.6173303e-2


def map_function(params):
    sim = BaseSimulation()
    sim.options = params["sim"].options

    # Calculate positions of atoms
    positions = []
    for i in range(params['n']):
        positions.append([params['d'] + i * params['d'], 0.0001, 0.0001])

    # Create the atoms and load them into the simulation
    chain = Atoms('Gd' + str(params['n']), positions=positions)
    sim.load_particles(chain)

    # Grab parameters from the dictionary
    q_m = params['q_m']
    c, o = params['c'], params['o']

    # Calculate the scattering vector
    q = q_m * 2 * np.pi / params['n']
    q_vec = np.array([q, 0., 0.])

    # Calculate dt and N from expected energy
    max_expected_energy = 4. * params['J'] * k_b_meV * o['spin']

    dt, N = calculate_dt_and_n(max_expected_energy * 0.01, max_expected_energy)
    print(max_expected_energy, dt, N)

    # Set relevant parameters on the sim
    params["N"] = N = N
    sim.options['dt'] = dt = dt
    sim.options['data_file'] = params["tmpdir"] + f'/datam_q{q}_n{params["n"]}_J{params["J"]}_T{params["T"]}_dt{dt}_S{params["S"]}_dt{dt}_d{params["d"]}_BZ{o["B"][2]}.h5'
    sim.options['transform_file'] = params["tmpdir"] + f'/transformsm_q{q}_n{params["n"]}_J{params["J"]}_T{params["T"]}_N{N}_dt{dt}_S{params["S"]}_dt{dt}_d{params["d"]}_BZ{o["B"][2]}.h5'

    # Set the initial positions of the spin to mirror a spin wave state
    for i in range(params["n"]):
        r, theta, phi = to_sph([np.cos(q * i), np.sin(q * i), 0.8])
        sim.particles.atoms[i].set_position(theta, phi)

    # Run the sim and transforms
    sim.run_simulation(N)
    sim.run_transformations(q_vec)

    # Close everything down so we can open it in the main thread.
    sim.close()

    # Return the options and q_vector used
    return sim.options, q_vec, q_m


class SpinChainTest(TestCase):
    def setUp(self):
        # Create temporary folder for data files
        self.tmpdir = mkdtemp()
        self.tmpdir = 'data/q_transform_test'

        # Initialize a simulation
        self.sim = BaseSimulation()

        # Some parameters
        self.n = n = 22  # Number of atoms
        self.d = d = 1.  # One Aangstroem between each atom
        self.J = J = 200.0
        self.T = T = 0.0
        self.S = S = 7/2

        # Configure the simulation
        self.sim.options['simulation_name'] = 'TestSpinChain'
        self.sim.options['l'] = 0
        self.sim.options['J'] = J * self.sim.constants['k_b']
        self.sim.options['T'] = T
        self.sim.options['B'] = [0., 0., 2.]
        self.sim.options['spin'] = S
        self.sim.options['pbc'] = (True, False, False)
        self.sim.options['debug'] = True

        # Calculate positions of atoms
        positions = []
        for i in range(n):
            positions.append([d + i * d, 0.0001, 0.0001])

        # Create the atoms and load them into the simulation
        chain = Atoms('Gd' + str(n), positions=positions)
        self.sim.load_particles(chain)

    def tearDown(self):
        self.sim.close()
        if not self.sim.options['debug']:
            rmtree(self.tmpdir)

    def test_linear_spin_waves(self):
        o, c = self.sim.options, self.sim.constants

        options, q_vec, q_m = map_function({
            "q_m": 2,
            "n": self.n,
            "c": c,
            "o": o,
            "J": self.J,
            "d": self.d,
            "T": self.T,
            "sim": self.sim,
            "S": self.S,
            "tmpdir": self.tmpdir
        })

        self.sim.options = options

        # Grab the data from the scans
        self.sim.run_simulation(1)

        self.sim.plot_components_individually('0', 'q_transforms_components.png')

        n = self.n
        q_vecs = [np.array([(x-n/2) * 2 * np.pi / n, 0, 0]) for x in range(n+1)]

        qs = np.array(range(101)) * 2 * np.pi / 100 - np.pi
        expected_energies = []
        for q in qs:
            # Calculate expected energy to overlay on the plot
            expected_energies.append(4. * self.J * k_b_meV * o['spin'] * (1. - np.cos(q)))

        fig, I_xx, hbar_omega = self.sim.plot_cross_section(q_vecs, max_energy=550, direction='x')

        plt.plot(qs, expected_energies, 'w--')

        figurename = f'I_of_omega_q{q_vec[0]}.png'
        fig.savefig(figurename, dpi=300)

        print('saved', figurename)

        # Reset the sim
        self.sim.close()
        self.sim.datafile = None
        self.sim.transformfile = None
        self.sim.transformtables = {}


if __name__ == '__main__':
    unittest.main()
