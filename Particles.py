# -*- coding: utf-8 -*-

"""
Class that contains all the particles in the simulation.
"""

from Particle import Particle
import pandas as pd
import math
from cli_helper import die

import ase.io

class Particles(object):
    def __init__(self, molecule, options, constants):
        self.options = options
        self.constants = constants
        self.atoms = []
        self.current = 0

        atoms = []

        for atom in molecule:
            if atom.symbol in options['magnetic_molecules']:
                atoms.append((len(atoms), atom))

        self.N_atoms = len(atoms)

        # Compute distances
        # Der er periodiske randbetingelser (Den helt til højre, er nabo til den helt til venstre).
        for id, atom in atoms:
            distances = []
            for inner_id, inner_atom in atoms:
                if id != inner_id:
                    distances.append((inner_id, molecule.get_distance(atom.index, inner_atom.index)))

            distances_frame = pd.DataFrame(distances)
            closest_neighbour_indexes = []

            if self.N_atoms > 2:
                distances_frame.sort_values(1, inplace=True)

                closest_atom = distances_frame.iloc[0]
                for i in range(0, self.N_atoms):
                    if math.fabs(distances_frame.iloc[i][1] - closest_atom[1]) < options['minimum_delta']:
                        closest_neighbour_indexes.append(distances_frame.iloc[i].name)
                    else:
                        break

            elif self.N_atoms > 1:
                # Special case, only two atoms, just add the other atom.
                closest_neighbour_indexes.append(distances[0][0])

            self.atoms.append(Particle(id, atom, self.N_atoms, closest_neighbour_indexes, self.options, self.constants))

            # TODO: Add support for initial conditions

    def __iter__(self):
        return self

    def next(self):
        if self.current > self.N_atoms - 1:
            self.current = 0
            raise StopIteration
        else:
            self.current += 1
            return self.atoms[self.current - 1]

    def __next__(self):
        return self.next()

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


def handle_molecule_from_file(options, constants):
    filetype = options['input_file'].split('.')[-1]
    allowed_formats = ['xyz', 'cube', 'pdb', 'traj', 'py']

    if not filetype in allowed_formats:
        die('Input filetype not supported. Supported types: {}'.format(allowed_formats))

    return Particles(ase.io.read(str(options['input_file'])), options, constants)
