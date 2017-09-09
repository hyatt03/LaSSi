"""
Class that contains all the particles in the simulation.
"""

from Particle import Particle
from openbabel import OBMolAtomIter

class Particles(object):
    def __init__(self, molecule, options):
        self.options = options
        self.atoms = []
        self.mol = molecule
        self.current = 0

        for obatom in OBMolAtomIter(molecule):
            self.atoms.append((len(self.atoms), obatom))

        self.N_atoms = len(self.atoms)

        for id, atom in self.atoms:
            self.atoms[id] = Particle(id, atom, self.N_atoms, self.options)

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
