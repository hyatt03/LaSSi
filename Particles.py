# -*- coding: utf-8 -*-

"""
Class that contains all the particles in the simulation.
"""

from Particle import Particle
from openbabel import OBMolAtomIter
import pandas as pd
import math

class Particles(object):
    def __init__(self, molecule, options):
        self.options = options
        self.atoms = []
        self.mol = molecule
        self.current = 0

        atoms = []

        for obatom in OBMolAtomIter(molecule):
            # TODO: We only want magnetic moelcules, but more than just gd
            if obatom.GetType() == 'Gd':
                atoms.append((len(atoms), obatom))

        self.N_atoms = len(atoms)

        # Compute distances
        # Der er periodiske randbetingelser (Den helt til højre, er nabo til den helt til venstre).
        for id, atom in atoms:
            distances = []
            for inner_id, inner_atom in atoms:
                if id != inner_id:
                    distances.append((inner_id, atom.GetDistance(inner_atom)))

            distances_frame = pd.DataFrame(distances)
            closest_neighbour_indexes = []

            if self.N_atoms > 1:
                distances_frame.sort_values(1, inplace=True)

                closest_atom = distances_frame.iloc[0]
                delta = 0.1
                for i in range(0, self.N_atoms):
                    if math.fabs(distances_frame.iloc[i][1] - closest_atom[1]) < delta:
                        closest_neighbour_indexes.append(distances_frame.iloc[i].name)
                    else:
                        break

            self.atoms.append(Particle(id, atom, self.N_atoms, closest_neighbour_indexes, self.options))

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

    # Calculate the effective B field for all atoms
    def combine_neighbours(self):
        for atom in self.atoms:
            atom.combine_neighbours(self.atoms)
