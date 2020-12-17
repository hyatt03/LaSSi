from SimulationBaseClass import BaseSimulation
from annealing import anneal_particles

from mpi4py import MPI

import ase.io, ase, ase.geometry

import numpy as np

import random, math


# Class for running simulations in parallel using MPI
# @TODO: There are potentially more methods that can be parallelized
# Evaluate which, and implement them
class MPIBaseSimulation(BaseSimulation):
    def __init__(self):
        # Initialize the base class
        super().__init__()

        # Initialize MPI
        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.mpi_cluster_size = self.mpi_comm.Get_size()
        self.mpi_tags = np.zeros((self.mpi_cluster_size,), dtype='i')
        self.is_master = self.mpi_rank == 0
        self.mpi_next_node = 0

        # print out a hello message from each node, so we can see the node works
        print(f'hello, im rank {self.mpi_rank + 1} of {self.mpi_cluster_size}')

        # Check the cluster is actually running
        assert(self.mpi_cluster_size > 1)

    def get_mpi_tag(self, dest):
        self.mpi_tags[dest] += 1
        return int(f'{dest+1}{self.mpi_tags[dest]}')

    def get_next_mpi_node(self):
        # We increment to discover which node is next
        self.mpi_next_node += 1

        # If the node does not exist, we reset
        if self.mpi_next_node == self.mpi_cluster_size:
            self.mpi_next_node = 1

        return self.mpi_next_node

    def load_particles(self, molecule=None):
        # If we are the master node, we want to load the particles
        if self.is_master:
            # We load the particle, either from a file or through the argument
            if molecule is not None:
                ase_molecule = molecule
            else:
                ase_molecule = ase.io.read(str(self.options['input_file']))

            # We break it down so we can transmit it
            symbols = []
            positions = []
            cell = ase_molecule.get_cell()
            for atom in ase_molecule:
                symbols.append(int(atom.number))
                positions.append(atom.position)

            # We send the cell, symbols, and positions
            for i in range(1, self.mpi_cluster_size):
                self.mpi_comm.send(symbols, dest=i, tag=self.get_mpi_tag(i))
                self.mpi_comm.send(positions, dest=i, tag=self.get_mpi_tag(i))
                self.mpi_comm.send(cell, dest=i, tag=self.get_mpi_tag(i))
        else:
            # We receive the symbols and positions
            symbols = self.mpi_comm.recv(source=0, tag=self.get_mpi_tag(self.mpi_rank))
            positions = self.mpi_comm.recv(source=0, tag=self.get_mpi_tag(self.mpi_rank))
            cell = self.mpi_comm.recv(source=0, tag=self.get_mpi_tag(self.mpi_rank))

        # Now we reconstruct it
        ase_atoms = []
        for symbol, position in zip(symbols, positions):
            ase_atoms.append(ase.Atom(symbol, position))

        # Using the reconstructed molecule, we initialize the "Particles" class
        super().load_particles(ase.Atoms(ase_atoms, cell=cell))

    def compute_positions_matrix(self):
        positions = np.empty((len(self.particles.atoms), 5))
        for idx, atom in enumerate(self.particles.atoms):
            c_pos = atom.pos  # Get cartesian position

            # Set the position of this atom in the matrix
            positions[idx, 0] = atom.theta
            positions[idx, 1] = atom.phi
            positions[idx, 2] = c_pos[0]
            positions[idx, 3] = c_pos[1]
            positions[idx, 4] = c_pos[2]

        return positions

    def mpi_send_positions_to_slaves(self, positions=None):
        if self.is_master:
            # Compute positions
            if positions is None:
                positions = self.compute_positions_matrix()

            # Next we send the updated position to all the nodes
            req = None
            for i in range(1, self.mpi_cluster_size):
                req = self.mpi_comm.Isend(positions, dest=i, tag=self.get_mpi_tag(i))

            # Wait for the last node to send
            if req is not None:
                req.wait()
        else:
            # Get the updated position from the master
            positions = np.empty((len(self.particles.atoms), 5))
            req = self.mpi_comm.Irecv(positions, source=0, tag=self.get_mpi_tag(self.mpi_rank))
            req.wait()

        # Update the positions
        for idx, atom in enumerate(self.particles.atoms):
            atom.pos = positions[idx, 2:]
            atom.theta = positions[idx, 0]
            atom.phi = positions[idx, 1]

    def mpi_sync_positions(self):
        # Initialize an empty array to hold all the positions
        positions = self.compute_positions_matrix()

        # We want to receive on the master node
        if self.is_master:
            # We create a dict that contains all the positions
            pos_dict = {}

            # We receive the data from the slave nodes
            req = None
            for i in range(1, self.mpi_cluster_size):
                pos_dict['positions_' + str(i)] = np.empty_like(positions)
                req = self.mpi_comm.Irecv(pos_dict['positions_' + str(i)], source=i, tag=self.get_mpi_tag(i))

            # Wait for the last node to send
            if req is not None:
                req.wait()

            # Next we want to compare our picture, with the picture from each node
            diff = np.zeros_like(positions)
            for i in range(1, self.mpi_cluster_size):
                # Find the difference between the nodes and our picture
                # and save it to the diff array
                diff += pos_dict['positions_' + str(i)] - positions

            # Now we update our position to reflect the changes from the nodes
            positions += diff
        else:
            # Send the position of this node to the master
            req = self.mpi_comm.Isend(positions, dest=0, tag=self.get_mpi_tag(self.mpi_rank))
            req.wait()

        # And we send the positons from the master to the slaves
        self.mpi_send_positions_to_slaves(positions)

    def run_anneal(self, steps):
        # Ensure the nodes are synced to begin with
        self.mpi_sync_positions()

        # Run the anneal on a single node
        if self.mpi_rank == 1:
            anneal_particles(self.options, self.constants, self.particles, steps)

        # Sync the results
        self.mpi_sync_positions()

    def run_simulation(self, iterations):
        # Check if the sim is ready to start
        self.check_sim_can_start()

        # Initialize variables
        o, c = self.options, self.constants
        sigma = math.sqrt(2. * o['l'] * c['k_b'] * o['T'] * c['hbar'] * o['dt'] /
                          ((c['g'] * c['mu_b']) ** 2. * o['spin']))

        # Compute what particles should be simulated where
        particle_map = {}
        my_particles = []
        if self.is_master:
            # Computed on the master node
            # Just runs through all the particles and assigns a node to each
            j = 1
            for i in range(len(self.particles.atoms)):
                particle_map[i] = j
                j += 1

                if j == self.mpi_cluster_size:
                    j = 1
        else:
            # Computed on the slaves
            # Checks if this particle should be simulated on here
            for i in range(len(self.particles.atoms)):
                if (i + self.mpi_rank - 1) / (self.mpi_cluster_size - 1) % 1 == 0:
                    my_particles.append(i)

        # If we have the master node, we want to open the datafile and find the starting position
        if self.is_master:
            self.open_datafile()

            # Count the number of rows
            i_0 = self.datafile.root.timeseries.p0.nrows

            # Send the count to all the nodes
            req = None
            for i in range(1, self.mpi_cluster_size):
                req = self.mpi_comm.isend(i_0, dest=i, tag=self.get_mpi_tag(i))

            # Wait for the last node to send
            if req is not None:
                req.wait()
        else:
            # Get the start of the iterator
            i_0 = self.mpi_comm.recv(source=0, tag=self.get_mpi_tag(self.mpi_rank))

        # Start simulation
        perc = 0
        i = i_0
        while i <= iterations:
            # ensure the effective B field is correct.
            self.particles.combine_neighbours()

            # We only want to compute the random field on the master
            if self.is_master:
                # Create a random perturbation to emulate temperature
                b_rand = random.gauss(0, sigma)
                u = random.random() * 2 * math.pi
                v = math.acos(2 * random.random() - 1)
                b_rand_sph = np.array([b_rand, u, v])

                # Send the random field to the nodes
                req = None
                for j in range(1, self.mpi_cluster_size):
                    req = self.mpi_comm.Isend(b_rand_sph, dest=j, tag=self.get_mpi_tag(j))

                # Now we want to compute the percentage and report it
                progress = int(100 * i / iterations)
                if o['debug'] and progress > perc:
                    perc = progress
                    print('Simulating {0}%'.format(perc))

                # Wait for the last node to send
                if req is not None:
                    req.wait()
            else:
                # Get the random field
                b_rand_sph = np.array([0., 0., 0.])
                self.mpi_comm.Recv(b_rand_sph, source=0, tag=self.get_mpi_tag(self.mpi_rank))

                # Take a step
                for p_id in my_particles:
                    # Grab the particle
                    particle = self.particles.atoms[p_id]

                    # Take a step
                    if o['integrator'] == 'ad_bs':
                        # Adams Bashforth method, 5th order,
                        # both numerically and energetically stable, but not great accuracy.
                        particle.ad_bs_step(b_rand_sph)
                    elif o['integrator'] == 'ad3':
                        particle.ad3_step(b_rand_sph)
                    elif o['integrator'] == 'RK4':
                        # Fourth order Runge Kutta, pretty common method, but not stable in energy
                        particle.take_rk4_step(b_rand_sph)
                    elif o['integrator'] == 'RK2':
                        # Also known as the midpoint method
                        particle.take_rk2_step(b_rand_sph)
                    elif o['integrator'] == 'euler':
                        particle.take_euler_step(b_rand_sph)
                    else:
                        raise ValueError('Invalid integrator, use ad_bs or RK4 in simulation options')

            # Sync positions from the steps taken
            self.mpi_sync_positions()

            # Next we want to save the new step to the sim file
            if self.is_master:
                for particle in self.particles.atoms:
                    # Figure out the tablename
                    tablename = 'p{}'.format(particle.id)

                    # Save the data to the buffer
                    self.datatables[tablename].append([(
                        i * o['dt'],  # t
                        particle.pos[0],  # x
                        particle.pos[1],  # y
                        particle.pos[2],  # z
                        0.0
                    )])

                    # Flush the data to file once in a while, increase to run faster (costs more memory)
                    if i % 1e7 == 0:
                        self.datatables[tablename].flush()

            i += 1

    def open_transformations_table(self):
        # Only complete on the master node
        if self.is_master:
            super().open_transformations_table()

    def run_transformations(self, q):
        # Only complete on the master node
        if self.is_master:
            super().run_transformations(q)

    def run_async_transformations(self, qs):
        # Only complete on the master node
        if self.is_master:
            super().run_async_transformations(qs)

    def plot_cross_section(self, q_vectors, max_energy=-1, direction='x'):
        # Only complete on the master node
        if self.is_master:
            super().plot_cross_section(q_vectors, max_energy, direction)

    def animate_spins_xyz(self, filename, every_nth=12, max_n_rows=4000):
        # Only complete on the master node
        if self.is_master:
            super().animate_spins_xyz(filename, every_nth, max_n_rows)

    def plot_components_individually(self, atomId, filename):
        # Only complete on the master node
        if self.is_master:
            super().plot_components_individually(atomId, filename)

    def plot_positions_xy(self, filename):
        # Only complete on the master node
        if self.is_master:
            super().plot_positions_xy(filename)

    def plot_spins_xy(self, filename):
        # Only complete on the master node
        if self.is_master:
            super().plot_spins_xy(filename)

    def plot_spins_xyz(self, filename):
        # Only complete on the master node
        if self.is_master:
            super().plot_spins_xyz(filename)

    def plot_single_spin_xyz(self, atomId, filename):
        # Only complete on the master node
        if self.is_master:
            super().plot_single_spin_xyz(atomId, filename)

    def plot_system_energies_as_f_of_t(self, filename):
        # Only complete on the master node
        if self.is_master:
            super().plot_system_energies_as_f_of_t(filename)

    def plot_energies(self, filename, energy_interval=None):
        # Only complete on the master node
        if self.is_master:
            super().plot_energies(filename, energy_interval)

    def plot_frequencies(self, filename, frequency_interval):
        # Only complete on the master node
        if self.is_master:
            super().plot_frequencies(filename, frequency_interval)
