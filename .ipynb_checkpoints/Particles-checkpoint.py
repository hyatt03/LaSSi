# -*- coding: utf-8 -*-

"""
Class that contains all the particles in the simulation.
"""

from Particle import Particle
import pandas as pd
import numpy as np
import math
from cli_helper import die

import ase.geometry, ase.io
from ase import Atom, Atoms

class Particles(object):
    def __init__(self, molecule, options, constants):
        self.options = options
        self.constants = constants
        self.atoms = []
        self.current = 0
        self.shape = ase.geometry.crystal_structure_from_cell(molecule.get_cell())
        self.ase = molecule

        if self.options['debug']:
            print('Loaded crystall with shape {} and {} atoms'.format(self.shape, len(molecule)))

        molecule = self.repeat_molecule(molecule)

        # Find dimensions of the molecule
        self.len_x, self.len_y, self.len_z = molecule.get_cell_lengths_and_angles()[:3]
        if all(molecule.get_cell_lengths_and_angles()[:3]) == 0.:
            self.len_x, self.len_y, self.len_z = self.find_cubic_size(molecule)

        atoms = []
        for atom in molecule:
            if atom.symbol in options['magnetic_molecules']:
                atoms.append((len(atoms), atom))

        self.N_atoms = len(atoms)

        # Compute distances to find closest neighbours
        # Also ensures periodic boundary conditions
        for id, atom in atoms:
            closest_neighbour_indexes = []
            if self.N_atoms > 1:
                # Start by determining the nearest neighbours for all the atoms we want to simulate
                distances = []
                for inner_id, inner_atom in atoms:
                    if id != inner_id:
                        distances.append((inner_id, molecule.get_distance(atom.index, inner_atom.index)))

                if self.N_atoms > 2:
                    # Next we want to enable periodic boundary conditions (if applicable)
                    close_sides = self.get_close_sides()

                    p = atom.position
                    for displace in close_sides:
                        # Create new atom for calculation
                        a = Atoms()
                        a += Atom(atom.symbol, (p[0] + displace[0], p[1] + displace[1], p[2] + displace[2]))

                        # Now go through all atoms in the first unit cell
                        for inner_id, inner_atom in atoms:
                            if id != inner_id:
                                a += inner_atom # Add the inner atom to compare
                                distances.append((inner_id, a.get_distance(0, 1)))
                                del a[1] # Remove it again

                    # Put the distances into a dataframe for easy processing
                    distances_frame = pd.DataFrame(distances)
                    distances_frame.sort_values(1, inplace=True)

                    # Compute the nearest neighbours
                    closest_atom = distances_frame.iloc[0]
                    for i in range(len(distances)):
                        neighbour_id = int(distances_frame.iloc[i][0])
                        if math.fabs(distances_frame.iloc[i][1] - closest_atom[1]) < options['minimum_delta'] and \
                                neighbour_id not in closest_neighbour_indexes:
                            closest_neighbour_indexes.append(neighbour_id)
                        else:
                            break

                else:
                    # Special case, only two atoms, just add the other atom.
                    closest_neighbour_indexes.append(distances[0][0])

            self.atoms.append(Particle(id, atom, self.N_atoms, closest_neighbour_indexes, self.options, self.constants))

    def __iter__(self):
        return self

    def __next__(self):
        if self.current > self.N_atoms - 1:
            self.current = 0
            raise StopIteration
        else:
            self.current += 1
            return self.atoms[self.current - 1]

    # Calculate the effective B field for all atoms
    def combine_neighbours(self):
        for atom in self.atoms:
            atom.combine_neighbours(self.atoms)

    def get_energy(self):
        energy = 0
        for atom in self.atoms:
            energy += atom.get_energy(self.atoms)

        return energy

    def get_atom_from_tablename(self, tablename):
        return self.atoms[int(tablename.replace('p', ''))]

    # Calculates a bounding box when this information is unavailable
    def find_cubic_size(self, molecule):
        # First set initial values and copy the positions
        x, y, z = 1, 1, 1
        positions = molecule.positions.copy()

        # We only want to calculate a size if more than one atom exists
        if len(positions) > 1:
            # Set initial guesses of the max and min coordinates
            max_s = positions[0].copy()
            min_s = positions[0].copy()

            # Set initial minimum distance between atoms
            min_d = [10, 10, 10]

            # Iterate over the positions
            for idx, p in enumerate(positions):
                # Check if we've found a new max or min for a given dimension
                for i in range(0, 3):
                    if p[i] > max_s[i]:
                        max_s[i] = p[i]
                    elif p[i] < min_s[i]:
                        min_s[i] = p[i]

                # Calculate the distances to the rest of the atoms and check if we've found a new minimum distance
                for idx2 in range(idx + 1, len(positions)):
                    p2 = positions[idx2]
                    dist = np.abs(p - p2)
                    for i in range(0, 3):
                        if dist[i] < min_d[i]:
                            min_d[i] = dist[i]

            # Calculate the size of the bounding box by subtracting the min position from the max position
            # Add the minimum distance between atoms to pad the bounding box
            x, y, z = max_s - min_s + min_d

        return x, y, z

    def repeat_molecule(self, molecule):
        repeats = self.options['repeat_cells']

        if repeats and len(repeats) > 0:
            try:
                molecule = molecule.repeat(repeats)

                if self.options['debug']:
                    print('Repeated molecule new size is: {} atoms'.format(len(molecule)))
            except ValueError:
                die('Could not copy unit cell, try using a .cif file instead.')

        return molecule

    def get_close_sides(self):
        close_sides = []

        # First check if we want periodic boundary conditions at all
        if self.options['pbc']:
            # Periodic boundary conditions for x
            if self.options['pbc'][0]:
                close_sides.append([self.len_x, 0, 0])
                close_sides.append([-self.len_x, 0, 0])

            # Periodic boundary conditions for y
            if self.options['pbc'][1]:
                close_sides.append([0, self.len_y, 0])
                close_sides.append([0, -self.len_y, 0])

            # Periodic boundary conditions for z
            if self.options['pbc'][2]:
                close_sides.append([0, 0, self.len_z])
                close_sides.append([0, 0, -self.len_z])

            # And all diagonals...
            # Start with x and y
            if self.options['pbc'][0] and self.options['pbc'][1]:
                close_sides.append([self.len_x, self.len_y, 0])
                close_sides.append([-self.len_x, self.len_y, 0])
                close_sides.append([self.len_x, -self.len_y, 0])
                close_sides.append([-self.len_x, -self.len_y, 0])

            # Diagonal periodic boundary conditions of y and z
            if self.options['pbc'][1] and self.options['pbc'][2]:
                close_sides.append([0, self.len_y, self.len_z])
                close_sides.append([0, -self.len_y, self.len_z])
                close_sides.append([0, self.len_y, -self.len_z])
                close_sides.append([0, -self.len_y, -self.len_z])

            # Diagonal periodic boundary conditions of z and x
            if self.options['pbc'][2] and self.options['pbc'][0]:
                close_sides.append([self.len_x, 0, self.len_z])
                close_sides.append([-self.len_x, 0, self.len_z])
                close_sides.append([self.len_x, 0, -self.len_z])
                close_sides.append([-self.len_x, 0, -self.len_z])

        return close_sides


def handle_molecule_from_file(options, constants):
    filetype = options['input_file'].split('.')[-1]
    allowed_formats = ['xyz', 'cif', 'cube', 'pdb', 'traj', 'py']

    if not filetype in allowed_formats:
        die('Input filetype not supported. Supported types: {}'.format(allowed_formats))

    return Particles(ase.io.read(str(options['input_file'])), options, constants)


def handle_molecule_from_ase(options, constants, molecule):
    return Particles(molecule, options, constants)
