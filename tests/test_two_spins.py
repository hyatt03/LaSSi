import unittest
from SimulationBaseClass import BaseSimulation
import tempfile
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt
from utils import to_sph


class TwoSpinTest(unittest.TestCase):
    def setUp(self):
        # Create temporary folder for data files
        self.tmpdir = tempfile.mkdtemp()

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
        self.sim.options['J'] = 173 * self.sim.constants['k_b']
        self.sim.options['T'] = 0
        self.sim.options['B'] = [0., 0., 0.]
        self.sim.options['debug'] = True

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

    def test_exchange_frequence_as_deviation_angle(self):
        deviation_angles = np.array(list(range(30, 45, 5))) * np.pi / 180

        for idx, d_angle in enumerate(deviation_angles):
            # Start by setting the initial position of the spins
            r, theta1, phi1 = to_sph([np.cos(d_angle), np.sin(d_angle), 0])
            r, theta2, phi2 = to_sph([-np.cos(d_angle), np.sin(d_angle), 0])

            self.sim.particles.atoms[0].set_position(theta1, phi1)
            self.sim.particles.atoms[1].set_position(theta2, phi2)

            # Next run the simulation
            self.sim.run_simulation(2 ** 24)

            # Run the transformation
            self.sim.run_transformations(np.array([0, 0, 0]))

            # max_value_xx = max(self.sim.transformtables['[0 0 0]'].cols.I_xx)
            # max_value_yy = max(self.sim.transformtables['[0 0 0]'].cols.I_yy)
            # max_value_zz = max(self.sim.transformtables['[0 0 0]'].cols.I_zz)
            #
            # for row in self.sim.transformtables['[0 0 0]'].where('I_xx == {}'.format(max_value_xx)):
            #     print('I_xx', max_value_xx, ', d_theta =', d_angle, ', E = ', row['energy'])
            #
            # for row in self.sim.transformtables['[0 0 0]'].where('I_yy == {}'.format(max_value_yy)):
            #     print('I_yy', max_value_yy, ', d_theta =', d_angle, ', E = ', row['energy'])
            #
            # for row in self.sim.transformtables['[0 0 0]'].where('I_zz == {}'.format(max_value_zz)):
            #     print('I_zz', max_value_yy, ', d_theta =', d_angle, ', E = ', row['energy'])

            self.sim.plot_energies('data/two_spins_test/t_two_spin_E_{}_' + str(d_angle) + '.png')
            plt.show()
            # self.sim.plot_spins_xyz('data/two_spins_test/two_spins_xyz_{}.png'.format(d_angle))

            # Reset the simulation for the next angle
            plt.close('all')
            self.sim.close()
            self.sim.datafile = None
            self.sim.transformfile = None
            self.sim.transformtables = {}
            rmtree(self.tmpdir)

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

        # max_value = max(self.sim.transformtables['[0 0 0]'].cols.I_xx)

        # The expected energy for larmor precession
        # expected_energy = - self.sim.constants['g'] * self.sim.constants['mu_b_meV'] * self.sim.options['B'][2]
        # max_value = max(self.sim.transformtables['[0 0 0]'].cols.I_xx)
        # for row in self.sim.transformtables['[0 0 0]'].where('I_xx == {}'.format(max_value)):
            # self.assertAlmostEqual(expected_energy, row['energy'], places=4)

if __name__ == '__main__':
    unittest.main()
