import unittest
from SimulationBaseClass import BaseSimulation
import tempfile
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt
from utils import to_sph, calculate_dt_and_n
from transformations import run_fft


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
        self.sim.options['debug'] = True

        self.sim.load_particles()

    def tearDown(self):
        self.sim.close()
        # rmtree(self.tmpdir)

    def test_exchange_frequence_as_deviation_angle(self):
        deviation_angles = np.array(list(range(0, 45))) * np.pi / 180
        deviation_angles = np.array([20 * np.pi / 180])

        for idx, d_angle in enumerate(deviation_angles):
            # dt and N
            expected_energy = 40
            dt, N = calculate_dt_and_n(expected_energy * 0.01, expected_energy * 2)
            self.sim.options['dt'] = dt

            # N = 200001

            self.sim.options['data_file'] = self.tmpdir + f'/data_n_{str(d_angle)}_{str(N)}.h5'
            self.sim.options['transform_file'] = self.tmpdir + f'/transforms_n_{str(d_angle)}_{str(N)}.h5'

            # Start by setting the initial position of the spins
            r, theta1, phi1 = to_sph([np.cos(d_angle), 0, np.sin(d_angle)])
            r, theta2, phi2 = to_sph([-np.cos(d_angle), 0, np.sin(d_angle)])

            self.sim.particles.atoms[0].set_position(theta1, phi1)
            self.sim.particles.atoms[1].set_position(theta2, phi2)

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

            # create an array containing times
            t = np.array(range(len(spin_1[0]))) * self.sim.options['dt']

            # plt.plot(t, spin_1[0], label = '$S_{1,x}$')
            # plt.plot(t, spin_1[1], label = '$S_{1,y}$')
            # plt.plot(t, spin_1[2], label = '$S_{1,z}$')
            # plt.plot(t, spin_2[0], label = '$S_{2,x}$')
            # plt.plot(t, spin_2[1], label = '$S_{2,y}$')
            # plt.plot(t, spin_2[2], label = '$S_{2,z}$')
            # plt.legend()
            # plt.show()

            plt.close()

            # Calculate the dot product of the spins as a function of time
            s1_d_s2 = np.array([np.dot(spin_1[:, i], spin_2[:, i]) for i in range(spin_1.shape[1])])

            # Run a fit of the simulation
            # p1, p2 = np.polyfit(t, s1_d_s2, 1)

            plt.plot(t, s1_d_s2, '.')
            # plt.plot(t, p1 * t + p2)

            plt.title('$S_1 \cdot S_2$')
            plt.xlabel('t [s]')
            plt.ylabel('$S_1 \cdot S_2$')

            plt.savefig('spin1_dot_spin2_long.png')
            plt.close()

            # Assert conservation of energy by checking the slope of the energy as a function of time
            # The energy is directly proportional to the dot product calculated earlier.
            # self.assertAlmostEqual(0.0, p1, 10)

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

            self.sim.plot_energies('data/two_spins_test/two_spin_E_{}_' + str(d_angle) + '.png', [0, 100])
            # plt.show()
            plt.close()

            self.sim.plot_spins_xyz('data/two_spins_test/two_spins_xyz_{}.png'.format(d_angle))
            # plt.show()
            plt.close()

            # for tablename, table in self.sim.datatables.items():
            for tablename, table in []:
                particle = self.sim.particles.get_atom_from_tablename(tablename)
                positions = [
                    list(table.cols.pos_x),
                    list(table.cols.pos_y),
                    list(table.cols.pos_z)
                ]

                X, Y, Z, freqs = run_fft([np.array(positions).reshape(-1, 3)], self.sim.options['dt'])

                energies = freqs * self.sim.constants['Hz_to_meV']

                plt.figure()
                plt.plot(energies, np.log(X), '.')
                plt.show()
                plt.figure()
                plt.plot(energies, np.log(Y), '.')
                plt.show()
                plt.figure()
                plt.plot(energies, np.log(Z), '.')
                # plt.xlim(0, 5)
                # plt.savefig(f'somethign_{particle.id}.png')
                plt.show()

            # Reset the simulation for the next angle
            plt.close('all')
            self.sim.close()
            self.sim.datafile = None
            self.sim.transformfile = None
            self.sim.transformtables = {}
            # rmtree(self.tmpdir)

        # Initialize the simulation, run it, and get initial states
        # self.sim.run_transformations(np.array([0, 0, 0]))
        # self.sim.plot_energies('two_spins_energy_{}.png')

        # max_value = max(self.sim.transformtables['[0 0 0]'].cols.I_xx)

        # The expected energy for larmor precession
        # expected_energy = - self.sim.constants['g'] * self.sim.constants['mu_b_meV'] * self.sim.options['B'][2]
        # max_value = max(self.sim.transformtables['[0 0 0]'].cols.I_xx)
        # for row in self.sim.transformtables['[0 0 0]'].where('I_xx == {}'.format(max_value)):
            # self.assertAlmostEqual(expected_energy, row['energy'], places=4)

if __name__ == '__main__':
    unittest.main()
