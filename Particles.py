# -*- coding: utf-8 -*-

"""
Class that contains all the particles in the simulation.
"""

from Particle import Particle
import pandas as pd
import math
from cli_helper import die

using_openbabel = False
try:
    import openbabel
    using_openbabel = True
except:
    import ase.io

class Particles(object):
    def __init__(self, molecule, options):
        self.options = options
        self.atoms = []
        self.current = 0

        atoms = []

        if options.using_openbabel:
            for obatom in openbabel.OBMolAtomIter(molecule):
                # TODO: We only want magnetic moelcules, but more than just gd
                if obatom.GetType() == 'Gd':
                    atoms.append((len(atoms), obatom))
        else:
            for atom in molecule:
                print(atom, atom.symbol)
                if atom.symbol == 'Gd':
                    atoms.append((len(atoms), atom))

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

def check_filetype(obConversion, filetype):
    for format in obConversion.GetSupportedInputFormat():
        if format.startswith(filetype):
            return True

    return False

def handle_molecule_from_file(options):
    options.using_openbabel = using_openbabel
    if options.using_openbabel:
        return handle_molecule_from_file_using_openbabel(options)
    else:
        return handle_molecule_from_file_using_ase(options)

def handle_molecule_from_file_using_openbabel(options):
    filetype = options.filename.split('.')[-1]
    obConversion = openbabel.OBConversion()

    if not check_filetype(obConversion, filetype):
        die('Input filetype not supported, check http://openbabel.org/docs/2.3.0/FileFormats/Overview.html for supported formats')

    obConversion.SetInAndOutFormats(str(filetype), "pdb")

    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, str(options.filename))

    print('Opened molecule with {} atoms and a total mass of {}'.format(mol.NumAtoms(), mol.GetExactMass()))

    return Particles(mol, options)

def handle_molecule_from_file_using_ase(options):
    filetype = options.filename.split('.')[-1]
    allowed_formats = ['xyz', 'cube', 'pdb', 'traj', 'py']

    if not filetype in allowed_formats:
        die('Input filetype not supported, try using openbabel')

    return Particles(ase.io.read(str(options.filename)), options)
