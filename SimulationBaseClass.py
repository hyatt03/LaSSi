#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('agg')
matplotlib.rcParams['agg.path.chunksize'] = 1000

from Particles import handle_molecule_from_file
from annealing import anneal_particles
from simulation_iterator import simulation_iterator
from TimeSeriesDescriptor import TimeSeriesDescriptor
from TransformsDescriptor import TransformsDescriptor
from fourier import transform_on_q

from tables import open_file, NaturalNameWarning
import numpy as np
import os
from pathlib import Path
import hashlib
import re

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # Not directly used, but required to use 3d projection

import warnings
warnings.filterwarnings('ignore', category=NaturalNameWarning)

scattering_regex = re.compile('^\[\s*([0-9]+).([0-9]*)\s*([0-9]+).([0-9]*)\s*([0-9]+).([0-9]*)\s*\]$')


class BaseSimulation(object):
    """Base simulation class, used as a starting point for simulations"""

    # Scientific constants
    constants = {
        'k_b': 1.38064852e-23,
        'g': -2.002,
        'mu_b': 9.274009994e-24,
        'hbar': 1.054571800e-34,
        'Hz_to_meV': 4.135665538536e-12,
        'mu_b_meV': 5.7883818012e-2
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
        'magnetic_molecules': ['Gd'],         # Selects the desired atoms for simulation
        'minimum_delta': 0.1                  # We work within the nearest neighbour approximation,
                                              # so we find all the atoms within 0.1 Å
    }

    datafile = None
    datatables = {}

    transformfile = None
    transformgroup = None
    transformtables = {}

    input_hash = ''

    particles = None

    def __init__(self):
        """Initiates the class with required constants and options."""
        self.constants['gamma'] = (self.constants['g'] * self.constants['mu_b']) / self.constants['hbar']

    def load_particles(self):
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
            self.datafile = open_file(str(d_path.resolve()), mode='r', title=datatitle)

            # Check the hash
            if self.datafile.root.metadata.hash[0].decode('UTF-8') != self.input_hash:
                raise ValueError('The input file used now does not match the input file used during simulation')

            # Load the data
            for table in self.datafile.root.timeseries:
                self.datatables[table.name] = table
        else:
            # Create the path
            os.makedirs(d_path.parent, exist_ok=True)
            self.datafile = open_file(str(d_path.resolve()), mode='a', title=datatitle)

            # Save the hash with the data
            meta = self.datafile.create_group('/', 'metadata', 'Meta data')
            self.datafile.create_array(meta, 'hash', obj=[self.input_hash])

            # Create group and table to store the data
            ts = self.datafile.create_group('/', 'timeseries', 'Time series')
            for particle in self.particles:
                tablename = 'p{}'.format(particle.id)
                tabledescription = 'Data table for particle {}'.format(particle.id)
                self.datatables[tablename] = self.datafile.create_table(ts, tablename, TimeSeriesDescriptor, tabledescription)

            # Start the simulation
            simulation_iterator(self.options, self.constants, self.particles, iterations, self.datatables)

    def run_transformations(self, q):
        # Verify required parameters are set
        if self.particles is None:
            raise ValueError('particles is not defined, please run load_particles first')

        if self.datatables is None:
            raise ValueError('Table is not defined, try running run_simulation!')

        if self.options['transform_file'] is None:
            raise ValueError('Need transform_file option in order to run a transform')

        if not type(q) is np.ndarray:
            raise ValueError('Expected q vector to be numpy array!')

        if not len(q) == 3:
            raise ValueError('Expected q vector to be 3 dimensional')

        # Open the transformfile
        if self.transformfile is None:
            t_path = Path(self.options['transform_file'])
            transformtitle = '{}_transform'.format(self.options['simulation_name'])
            if t_path.is_file():
                # Load the file
                self.transformfile = open_file(str(t_path.resolve()), mode='a', title=transformtitle)

                # Load the group
                self.transformgroup = self.transformfile.root.transforms

                # Load the tables
                for table in self.transformgroup:
                    self.transformtables[table.name] = table
            else:
                # Create the file
                os.makedirs(t_path.parent, exist_ok=True)
                self.transformfile = open_file(str(t_path.resolve()), mode='a', title=transformtitle)

                # Create group
                self.transformgroup = self.transformfile.create_group('/', 'transforms', 'Scattering cross sections')

        # Get or compute the transforms
        if str(q) in self.transformtables:
            return self.transformtables[str(q)]

        # Create the table
        tabledescription = 'Cross section for q = {}'.format(str(q))
        self.transformtables[str(q)] = self.transformfile.create_table(self.transformgroup, str(q), TransformsDescriptor, tabledescription)

        [I_aa_temp, energies, frequencies] = transform_on_q(q, self.options, self.constants, self.datatables, self.particles)

        row = self.transformtables[str(q)].row
        for rowIndex in range(0, len(frequencies)):
            row['energy'] = energies[rowIndex]
            row['frequency'] = frequencies[rowIndex]
            row['I_xx'] = np.abs(I_aa_temp[0][rowIndex])
            row['I_yy'] = np.abs(I_aa_temp[1][rowIndex])
            row['I_zz'] = np.abs(I_aa_temp[2][rowIndex])
            row.append()

        self.transformtables[str(q)].flush()
        return self.transformtables[str(q)]

    def plot_spins_xy(self, filename):
        fig, ax = plt.subplots()

        for key, value in self.datatables.items():
            ax.plot(value.cols.pos_x, value.cols.pos_y)

        plt.xlabel('$S_x$')
        plt.ylabel('$S_y$')

        plt.axis('equal')
        plt.savefig(filename, bbox_inches='tight', dpi=300)

    def plot_spins_xyz(self, filename):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for key, value in self.datatables.items():
            ax.plot(value.cols.pos_x, value.cols.pos_y, value.cols.pos_z)

        ax.set_xlabel('$S_x$')
        ax.set_ylabel('$S_y$')
        ax.set_zlabel('$S_z$')

        plt.axis('equal')
        plt.savefig(filename, bbox_inches='tight', dpi=300)

    def plot_energies(self, filename, energy_interval):
        for direction in ['xx', 'yy', 'zz']:
            fig, ax = plt.subplots()

            for key, value in self.transformtables.items():
                ax.plot(value.cols.energy, value.cols._f_col('I_{}'.format(direction)))

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

    def plot_scattering_cross_section(self, filename):
        cmap = plt.get_cmap('PiYG')

        for direction in ['xx', 'yy', 'zz']:
            x = []
            y = []
            z = []
            for key, value in self.transformtables.items():
                res = scattering_regex.findall(key)
                if len(res) > 0: # There should be max 1
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

            z = z[:-1, :-1]
            levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

            fig, ax = plt.subplots()

            im = ax.pcolormesh(x, y, z, cmap=cmap, norm=norm)
            fig.colorbar(im, ax=ax)

            plt.savefig(filename.format(direction), bbox_inches='tight', dpi=300)


    def close(self):
        # Close the datafile
        if self.datafile is not None:
            self.datafile.close()

        # Close the transformfile
        if self.transformfile is not None:
            self.transformfile.close()
