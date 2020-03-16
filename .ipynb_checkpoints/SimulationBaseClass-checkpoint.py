#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib

matplotlib.use('Qt5Agg')
# matplotlib.rcParams['agg.path.chunksize'] = 1000

from Particles import handle_molecule_from_file, handle_molecule_from_ase
from annealing import anneal_particles
from simulation_iterator import simulation_iterator
from TimeSeriesDescriptor import TimeSeriesDescriptor
from TransformsDescriptor import TransformsDescriptor
from fourier import transform_on_q

from tables import open_file, NaturalNameWarning, Filters
import numpy as np
import os
from pathlib import Path
import hashlib
import re
import tempfile
import multiprocessing

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Not directly used, but required to use 3d projection

try:
    from PIL import Image
except:
    pass

import warnings

scattering_regex = re.compile('^\[\s*([0-9]+).([0-9]*)\s*([0-9]+).([0-9]*)\s*([0-9]+).([0-9]*)\s*\]$')
hdf_file_filter = Filters(complevel=1)


def run_transform_from_dict(params):
    q, options, constants, datatables, particles, fourier_length = params['q'], params['options'], \
                                                                   params['constants'], params['datatables'], \
                                                                   params['particles'], params['fourier_length']

    if params['run_transform']:
        I_aa_temp, energies, frequencies = transform_on_q(q, options, constants, datatables, particles, fourier_length)
        return I_aa_temp, energies, frequencies, params

    # In case the transform already exists
    return None, None, None, params


class BaseSimulation(object):
    """Base simulation class, used as a starting point for simulations"""

    # Scientific constants
    constants = {
        'k_b': 1.38064852e-23,
        'g': -2.002,
        'mu_b': 9.274009994e-24,
        'hbar': 1.054571800e-34,
        'Hz_to_meV': 4.135665538536e-12,
        'mu_b_meV': 5.7883818012e-2,
        'gamma': -1.760859644e11
    }

    # Simulation options
    options = {
        'simulation_name': 'BaseSimulation',  # Name of the simulation
        'input_file': None,                   # Defines the crystal structure
        'data_file': None,                    # Defines where to save the data, if it already exists,
                                              # it simply loads the data
        'transform_file': None,               # Where to save the fourier transforms
        'spin': 0,                            # Set the spin of the particle we're simulating
        'l': 5e-4,                            # The dampening of the system
        'T': 0,                               # Temperature of the system
        'J': 0,                               # Nearest neighbour interaction constant
        'B': np.array([0, 0, 0]),             # External B-field
        'dt': 1e-15,                          # Small time interval
        'magnetic_molecules': ['Gd', 'Ni'],   # Selects the desired atoms for simulation
        'minimum_delta': 0.1,                 # We work within the nearest neighbour approximation,
                                              # so we find all the atoms within 0.1 Å
        'repeat_cells': None,                 # Support for multiple unit cells, specify how many repeats along the
                                              # three unit vectors of the structure
        'pbc': None,                          # Support for periodic boundary,
                                              # specify which directions should be periodic
        'debug': False,                       # Select if debugging output is wanted
        'anneal_T' : None,
        'integrator': 'ad_bs'
    }

    datafile = None
    datatables = {}

    transformfile = None
    transformgroup = None
    transformtables = {}

    input_hash = ''

    particles = None

    def __init__(self):
        pass

    def load_particles(self, molecule=None):
        if molecule is not None:
            self.particles = handle_molecule_from_ase(self.options, self.constants, molecule)
        else:
            if self.options['input_file'] is None:
                raise ValueError('Need input_file option in order to load a particle')

            # Start by acquiring the hash of the file, to prevent errors due to file changes
            hasher = hashlib.md5()
            with open(self.options['input_file'], 'rb') as inp:
                buf = inp.read()
                hasher.update(buf)

            self.input_hash = hasher.hexdigest()

            # Load the file
            self.particles = handle_molecule_from_file(self.options, self.constants)

    def run_anneal(self, steps):
        anneal_particles(self.options, self.constants, self.particles, steps)

    def run_simulation(self, iterations):
        if self.particles is None:
            raise ValueError('particles is not defined, please run load_particles first')

        if self.options['data_file'] is None:
            raise ValueError('Need data_file option in order to run a simulation')

        d_path = Path(self.options['data_file'])
        datatitle = '{}_data'.format(self.options['simulation_name'])

        if d_path.is_file():
            # Load the file
            self.datafile = open_file(str(d_path.resolve()), mode='a', title=datatitle, filters=hdf_file_filter)

            # Check the hash
            if self.datafile.root.metadata.hash[0].decode('UTF-8') != self.input_hash:
                raise ValueError('The input file used now does not match the input file used during simulation')

            # Load the data
            for table in self.datafile.root.timeseries:
                self.datatables[table.name] = table

            # Count the number of rows
            i_0 = self.datafile.root.timeseries.p0.nrows

            # We want to keep going if the number of rows is less than the number of iterations
            if i_0 < iterations:
                simulation_iterator(self.options, self.constants, self.particles, iterations, self.datatables, i_0)

        else:
            # Create the path
            os.makedirs(d_path.parent, exist_ok=True)
            self.datafile = open_file(str(d_path.resolve()), mode='a', title=datatitle, filters=hdf_file_filter)

            # Save the hash with the data
            meta = self.datafile.create_group('/', 'metadata', 'Meta data')
            self.datafile.create_array(meta, 'hash', obj=[self.input_hash])

            # Create group and table to store the data
            ts = self.datafile.create_group('/', 'timeseries', 'Time series')
            for particle in self.particles:
                tablename = 'p{}'.format(particle.id)
                tabledescription = 'Data table for particle {}'.format(particle.id)
                self.datatables[tablename] = self.datafile.create_table(ts, tablename, TimeSeriesDescriptor,
                                                                        tabledescription)

            # Start the simulation
            simulation_iterator(self.options, self.constants, self.particles, iterations, self.datatables)

    # Normalizes numpy array to values between 0 and 1
    def normalize_intensity(self, I):
        # Convert to numpy array for convenience
        I = np.array(I)

        # We subtract the mean so everything has a zero mean
        I -= np.mean(I)

        # We add the minimum value so I is >=0
        I += np.min(I)

        # Avoid divide by zero error
        # We just devide by a small number
        # It's most likely that all the values simply are 0
        if np.max(I) != 0.:
            # We divide by the max value so we get I<=1
            I /= np.max(I)
        else:
            I /= 0.01

        # I is now normalized to be in [0; 1]
        return I

    def open_transformations_table(self):
        # Verify required parameters are set
        if self.particles is None:
            raise ValueError('particles is not defined, please run load_particles first')

        if self.datatables is None:
            raise ValueError('Table is not defined, try running run_simulation!')

        if self.options['transform_file'] is None:
            raise ValueError('Need transform_file option in order to run a transform')

        # Open the transformfile
        if self.transformfile is None:
            t_path = Path(self.options['transform_file'])
            transformtitle = '{}_transform'.format(self.options['simulation_name'])
            if t_path.is_file():
                # Load the file
                self.transformfile = open_file(str(t_path.resolve()), mode='a', title=transformtitle, filters=hdf_file_filter)

                # Load the group
                self.transformgroup = self.transformfile.root.transforms

                # Load the tables
                for table in self.transformgroup:
                    self.transformtables[table.name] = table
            else:
                # Create the file
                os.makedirs(t_path.parent, exist_ok=True)
                self.transformfile = open_file(str(t_path.resolve()), mode='a', title=transformtitle, filters=hdf_file_filter)

                # Create group
                self.transformgroup = self.transformfile.create_group('/', 'transforms', 'Scattering cross sections')

    def run_transformations(self, q):
        if not type(q) is np.ndarray:
            raise ValueError('Expected q vector to be numpy array!')

        if not len(q) == 3:
            raise ValueError('Expected q vector to be 3 dimensional')

        self.open_transformations_table()

        # Get or compute the transforms
        if str(q) in self.transformtables:
            return self.transformtables[str(q)]

        # Calculate the fourier length
        data_length = len(self.datatables['p0'].cols.pos_x)
        fourier_length = int(2 ** np.ceil(np.log2(data_length)))

        # Create the table
        tabledescription = 'Cross section for q = {}'.format(str(q))

        # Ignore natural name warning, we don't care
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NaturalNameWarning)
            self.transformtables[str(q)] = self.transformfile.create_table(self.transformgroup, str(q), TransformsDescriptor, tabledescription)

        # Run the actual transform
        I_aa_temp, energies, frequencies = transform_on_q(q, self.options, self.constants, self.datatables, self.particles, fourier_length)

        # Normalize the intensities so we can compare them
        for idx in range(len(I_aa_temp)):
            I_aa_temp[idx] = self.normalize_intensity(I_aa_temp[idx])

        buffer = []
        for rowIndex in range(0, len(I_aa_temp[0])):
            buffer.append((energies[rowIndex],
                           frequencies[rowIndex],
                           np.abs(I_aa_temp[0][rowIndex]),
                           np.abs(I_aa_temp[1][rowIndex]),
                           np.abs(I_aa_temp[2][rowIndex])))

        self.transformtables[str(q)].append(buffer)
        self.transformtables[str(q)].flush()

        return self.transformtables[str(q)]

    def run_async_transformations(self, qs):
        self.open_transformations_table()

        parameter_sets = []

        # Calculate the fourier length
        data_length = len(self.datatables['p0'].cols.pos_x)
        fourier_length = int(2 ** np.ceil(np.log2(data_length)))

        data = {}
        for key in self.datatables:
            data[key] = np.array([
                self.datatables[key].cols.pos_x,
                self.datatables[key].cols.pos_y,
                self.datatables[key].cols.pos_z
            ], dtype=np.complex128)

        for q in qs:
            if not type(q) is np.ndarray:
                raise ValueError('Expected q vector to be numpy array!')

            if not len(q) == 3:
                raise ValueError('Expected q vector to be 3 dimensional')

            parameter_sets.append({
                "q": q,
                "options": self.options,
                "constants": self.constants,
                "datatables": data,
                "particles": self.particles,
                "fourier_length": fourier_length,
                "run_transform": not str(q) in self.transformtables
            })

        results_tables = []

        p = multiprocessing.Pool(8)
        for I_aa_temp, energies, frequencies, params in p.map(run_transform_from_dict, parameter_sets):
            q_str = str(params['q'])
            if I_aa_temp is None:
                results_tables.append(self.transformtables[q_str])
            else:
                # Normalize the intensities so we can compare them
                for idx in range(len(I_aa_temp)):
                    I_aa_temp[idx] = self.normalize_intensity(I_aa_temp[idx])

                # Create the table
                tabledescription = 'Cross section for q = {}'.format(q_str)

                # Ignore natural name warning, we don't care
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", NaturalNameWarning)
                    self.transformtables[q_str] = self.transformfile.create_table(self.transformgroup, q_str,
                                                                                   TransformsDescriptor, tabledescription)

                buffer = []
                for rowIndex in range(0, len(I_aa_temp[0])):
                    buffer.append((energies[rowIndex],
                                   frequencies[rowIndex],
                                   np.abs(I_aa_temp[0][rowIndex]),
                                   np.abs(I_aa_temp[1][rowIndex]),
                                   np.abs(I_aa_temp[2][rowIndex])))

                self.transformtables[q_str].append(buffer)
                self.transformtables[q_str].flush()

                results_tables.append(self.transformtables[q_str])

        return results_tables

    def plot_cross_section(self, q_vectors, max_energy=-1, direction='x'):
        if len(q_vectors) < 1:
            raise ValueError('q_vectors needs to be an array of vectors.')

        if direction not in ['x', 'y', 'z']:
            raise ValueError('direction needs to be x, y, or z.')

        direction_number = {"x": 0, "y": 1, "z": 2}[direction]

        q_vectors_to_calculate = []

        # We want to ensure all the q vectors are calculated, and we want to calculate them asynchronously
        for q in q_vectors:
            if str(q) not in self.transformtables:
                q_vectors_to_calculate.append(q)

        # Calculate all the missing q vectors
        self.run_async_transformations(q_vectors_to_calculate)

        # Figure out what index has the largest allowed energy
        # If it's none of them, we just use the final element
        last_idx = -1
        energies = self.run_transformations(q_vectors[0]).cols.energy
        for idx, energy in enumerate(energies):
            if energy > max_energy and max_energy > 0:
                last_idx = idx
                break

        # Resize the energies and make it a numpy array
        hbar_omega = np.array(energies[:last_idx]).reshape((-1, 1))

        # Initialize empty arrays to contain image data
        I_of_omega = np.zeros((hbar_omega.shape[0], len(q_vectors)))

        # Grab the image data
        for idx, q in enumerate(q_vectors):
            # Get the relevant table
            transformation_table = self.run_transformations(q)

            # Grab the intensities
            I_of_omega[:, idx] = np.array(transformation_table.col(f'I_{direction}{direction}')[:last_idx])

        # Flip the array, so it's the right way around.
        I_of_omega = np.flipud(I_of_omega)

        # Construct the plot
        plt.figure(figsize=(10, 9), tight_layout=True)
        extent = [q_vectors[0][direction_number], q_vectors[-1][direction_number], hbar_omega[1, 0], hbar_omega[-1, 0]]
        im = plt.imshow(I_of_omega, extent=extent, interpolation='nearest')
        ax = plt.gca()
        ax.set_aspect(abs(extent[1] - extent[0]) / abs(extent[3] - extent[2]))
        cbar = ax.figure.colorbar(im, ax=ax)
        im.set_clim(vmin=0.0, vmax=1.0)

        # Prettify
        # Start with the labels
        fontsize = 18
        q_labels = ['0', '0', '0']
        q_labels[direction_number] = 'X'
        plt.xlabel(f'$q = ({q_labels[0]}, {q_labels[1]}, {q_labels[2]}) [Å^{"{-1}"}]$', fontsize=fontsize)
        plt.ylabel('Energy [meV]', fontsize=fontsize)
        cbar.set_label('Normalized intensity [A.U.]', fontsize=fontsize)

        # Now the ticks
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        cbar.set_ticks([])

        # return the figure and the data
        # The user can show the figure with fig.show()
        return plt.gcf(), hbar_omega, I_of_omega


    def animate_spins_xyz(self, filename, every_nth=12, max_n_rows=4000):
        if not Image:
            print('animations not supported without pillow')

        print('animating')
        frames = []
        datatables = []
        for key, value in self.datatables.items():
            datatables.append(value)

        n = datatables[0].nrows
        if n > max_n_rows:
            n = max_n_rows

        for i in range(n):
            if i % every_nth == 0:
                print(i / datatables[0].nrows)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                time = 0
                for idx, t in enumerate(datatables):
                    x_0 = self.particles.atoms[idx].lattice_position[0]
                    y_0 = self.particles.atoms[idx].lattice_position[1]
                    z_0 = self.particles.atoms[idx].lattice_position[2]

                    row = t.read(i, i + 1)[0]
                    x = [row[1] + x_0, x_0]
                    y = [row[2] + y_0, y_0]
                    z = [row[3] + z_0, z_0]

                    ax.plot(x, y, z, 'b-')

                    if idx == 0:
                        time = row[4]

                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
                ax.set_zlim(-2, 2)
                ax.set_xlabel('$S_x$')
                ax.set_ylabel('$S_y$')
                ax.set_zlabel('$S_z$')
                plt.title('t = {}'.format(time))

                # Create a temporary file and save the plot to it
                # Then we open it as an image and append it to frames
                # Slightly hacky way to implement animations.
                fp = tempfile.TemporaryFile()
                plt.savefig(fp)
                plt.close()
                frames.append(Image.open(fp))

        frames[0].save(filename, save_all=True, append_images=frames[1:])

        # Cleanup
        for f in frames:
            f.close()

    def plot_positions_xy(self, filename):
        fig, ax = plt.subplots()

        x, y = [], []
        for p in self.particles.atoms:
            x.append(p.lattice_position[0])
            y.append(p.lattice_position[1])

        ax.plot(x, y, 'o')

        plt.xlabel('$X$')
        plt.ylabel('$Y$')

        plt.axis('equal')
        plt.savefig(filename, bbox_inches='tight', dpi=300)

    def plot_spins_xy(self, filename):
        fig, ax = plt.subplots()

        i = 0
        for key, value in self.datatables.items():
            x_0 = self.particles.atoms[i].lattice_position[0]
            y_0 = self.particles.atoms[i].lattice_position[1]

            j = 0
            for row in value.iterrows():
                # if j>1000:
                #     break

                ax.plot(row['pos_x'] + x_0, row['pos_y'] + y_0, 'b.')
                j += 1

            i += 1

        plt.xlabel('$S_x$')
        plt.ylabel('$S_y$')

        # plt.axis('equal')
        plt.savefig(filename, bbox_inches='tight', dpi=300)

    def plot_spins_xyz(self, filename):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        i = 0
        for key, value in self.datatables.items():
            x_0 = self.particles.atoms[i].lattice_position[0]
            y_0 = self.particles.atoms[i].lattice_position[1]
            z_0 = self.particles.atoms[i].lattice_position[2]

            x, y, z = [], [], []

            for row in value.iterrows():
                x.append(row['pos_x'] + x_0)
                y.append(row['pos_y'] + y_0)
                z.append(row['pos_z'] + z_0)

            ax.plot(x, y, z, 'b.')
            i += 1

        ax.set_xlabel('$S_x$')
        ax.set_ylabel('$S_y$')
        ax.set_zlabel('$S_z$')

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)

        # plt.axis('equal')
        plt.savefig(filename, bbox_inches='tight', dpi=300)

    def plot_single_spin_xyz(self, atomId, filename):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x, y, z = [], [], []

        for row in self.datatables['p' + str(atomId)].iterrows():
            x.append(row['pos_x'])
            y.append(row['pos_y'])
            z.append(row['pos_z'])

        ax.plot(x, y, z, 'b.')
        ax.set_xlabel('$S_x$')
        ax.set_ylabel('$S_y$')
        ax.set_zlabel('$S_z$')

        # plt.axis('equal')
        plt.savefig(filename, bbox_inches='tight', dpi=300)

    def plot_system_energies_as_f_of_t(self, filename):
        time = []
        E = []

        datatables = []
        for key, value in self.datatables.items():
            datatables.append(value)

        for i in range(datatables[0].nrows):
            E_ = 0
            for idx, t in enumerate(datatables):
                row = t.read(i, i + 1)[0]
                E_ += row[0]

                if idx == 0:
                    time.append(row[4])

            E.append(E_)

        fig, ax = plt.subplots()

        ax.plot(time, E, '-')

        plt.xlabel('time')
        plt.ylabel('Total energy of system')
        plt.savefig(filename, bbox_inches='tight', dpi=300)

    def plot_energies(self, filename, energy_interval=None):
        for direction in ['xx', 'yy', 'zz']:
            fig, ax = plt.subplots()

            for key, value in self.transformtables.items():
                ax.plot(value.cols.energy, value.cols._f_col('I_{}'.format(direction)))

            if energy_interval is not None:
                plt.xlim(energy_interval[0], energy_interval[1])
            plt.xlabel('Energy [meV]')
            plt.ylabel('Intensity [A.U.]')
            plt.savefig(filename.format(direction), bbox_inches='tight', dpi=300)

    def plot_frequencies(self, filename, frequency_interval):
        for direction in ['xx', 'yy', 'zz']:
            fig, ax = plt.subplots()

            for key, value in self.transformtables.items():
                ax.plot(value.cols.frequency, value.cols._f_col('I_{}'.format(direction)))

            plt.xlim(frequency_interval[0], frequency_interval[1])

            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Intensity [A.U.]')
            plt.savefig(filename.format(direction), bbox_inches='tight', dpi=300)

    def plot_qxx_vs_qyy(self, filename):
        for key, value in self.transformtables.items():
            print(key)

    def plot_scattering_cross_section(self, filename):
        cmap = plt.get_cmap('PiYG')

        for direction in ['xx', 'yy', 'zz']:
            x = []
            y = []
            z = []
            for key, value in self.transformtables.items():
                res = scattering_regex.findall(key)
                if len(res) > 0:  # There should be max 1
                    q_x = float('{}.{}'.format(res[0][0], res[0][1]))
                    q_y = float('{}.{}'.format(res[0][2], res[0][3]))
                    q_z = float('{}.{}'.format(res[0][4], res[0][5]))

                    # Calculate the magnitude of the scattering vector
                    x.append(np.sqrt(q_x ** 2 + q_y ** 2 + q_z ** 2))

                    # Create energies array, should be constant across all qs
                    if len(y) < 1:
                        y = value.cols.energy

                    # Append intensities for this combo of scattering vector and energies
                    z.append(value.cols._f_col('I_{}'.format(direction)))

            x = np.array(x)
            y = np.array(y)
            z = np.array(z)

            # print('x', x)
            # print('y', y)
            # print('z', z)

            # print(z)

            # print(x.shape)
            # print(y.shape)
            # print(z.shape)

            # z_min, z_max = -np.abs(z).max(), np.abs(z).max()
            # fig, ax = plt.subplots()
            # c = ax.pcolorfast(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
            # ax.set_title('pcolorfast')
            # fig.colorbar(c, ax=ax)

            minimum_E = 0
            maximum_E = 0.6
            cut = 22

            fig, ax = plt.subplots()
            im = plt.imshow(np.transpose(np.log10(z)), aspect='auto', interpolation='none')
            im.set_cmap('nipy_spectral')
            cbar = plt.colorbar(im, orientation="vertical")
            cbar.set_label(r'log(Intensity) [a.u.]', fontsize=20)
            # c = ax.pcolormesh(x, y, np.transpose(z))
            # plt.xlim(0, 12.5)
            # plt.ylim(0, 0.6)
            # fig.colorbar(c, ax=ax)

            plt.savefig(filename.format(direction), bbox_inches='tight', dpi=300)

            # z = z[:-1, :-1]
            # levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())
            # norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

            # fig, ax = plt.subplots()

            # im = ax.pcolormesh(x, y, z, cmap=cmap, norm=norm)
            # fig.colorbar(im, ax=ax)

            # plt.savefig(filename.format(direction), bbox_inches='tight', dpi=300)

    def close(self):
        # Close the datafile
        if self.datafile is not None:
            self.datafile.close()

        # Close the transformfile
        if self.transformfile is not None:
            self.transformfile.close()
