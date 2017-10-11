#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import openbabel
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from optparse import OptionParser
from Particles import Particles
from simulation_iterator import simulation_iterator
from annealing import anneal_particles

def die(message = ''):
    print('An error occurred')
    if message:
        print(message)

    sys.exit(1)

def die_prompt(message=''):
    print(message)
    res = raw_input('Type y for exit: ')
    if res == 'y':
        sys.exit(1)

def handle_arguments():
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename", help="get crystal structure from FILE", metavar="FILE")
    parser.add_option("-s", "--spin", dest="spin", help="Set the simulated spin", metavar="SPIN")
    parser.add_option("-l", "--dampening", dest="l", help="(OPTIONAL) Set the dampening factor", metavar="DAMP")
    parser.add_option("-T", "--temperature", dest="T", help="Set the temperature", metavar="TEMP")
    parser.add_option("-B", "--magneticfield", dest="B", help="Set the external B field, comma delimited (-B x,y,z)", metavar="MAGN")
    parser.add_option("-J", "--NNIC", dest="J", help="Set the nearest neighbour interaction constant", metavar="NNIC") # I enheder af kelvin.
    parser.add_option("-N", "--iterations", dest="N_simulation", help="Set the amount of iterations", metavar="ITER")
    parser.add_option("-A", "--anneal", dest="anneal", help="Enable annealing in N steps", metavar="A")
    parser.add_option("-p", "--plot", dest="should_plot", help="Do you want to plot the positions?", metavar="P")
    parser.add_option("--dt", dest="dt", help="Set the dt", metavar="DT")

    (options, args) = parser.parse_args()

    if not options.filename:
        die('An inputfile is required!')

    denominator = 1
    if options.spin:
        if '/' in options.spin:
            denominator = float(options.spin.split('/')[1])

        options.spin = float(options.spin.split('/')[0]) / denominator

    if not options.spin or options.spin < 2:
        die_prompt('This simulation uses semiclassical approximations that are best for spin 2 or higher! Are you sure you want to continue?')

    if options.l:
        options.l = float(options.l)
    else:
        options.l = 5e-4

    if options.T:
        options.T = float(options.T)
    else:
        die('You must set the temperature!')

    if options.J:
        options.J = float(options.J)
    else:
        die('You must the the nearest neighbour inteaction constant (J)')

    if options.B and ',' in options.B:
        B = options.B.split(',')
        options.B = np.array([0, 0, 0], dtype='float64')

        if B[0]:
            options.B[0] = float(B[0])

        if B[1]:
            options.B[1] = float(B[1])

        if B[2]:
            options.B[2] = float(B[2])

    else:
        print('No external B field applied')
        options.B = np.array([0, 0, 0])

    if options.N_simulation:
        options.N_simulation = int(options.N_simulation)
    else:
        die('You must set the iterations (N_simulation)!')

    if options.dt:
        options.dt = float(options.dt)
    else:
        die('You must set the dt (for example 4.14e-15)!')

    if options.anneal:
        options.anneal = int(float(options.anneal))

    return options

def handle_constant_properties(options):
    # Set data directory
    # TODO: Optional parameter for data dir.
    options.data_dir = os.path.dirname(os.path.abspath(__file__)) + '/data'

    # Ensure data dir exists
    if not os.path.isdir(options.data_dir):
        os.makedirs(options.data_dir)

    """
    Physical quantities
    """
    options.k_b = 1.38064852e-23
    options.g = -2.002
    options.mu_b = 9.274009994e-24
    options.hbar = 1.054571800e-34
    options.gamma = options.g * options.mu_b / options.hbar

    return options

def check_filetype(obConversion, filetype):
    for format in obConversion.GetSupportedInputFormat():
        if format.startswith(filetype):
            return True

    return False

def handle_molecule_from_file(options):
    filetype = options.filename.split('.')[-1]
    obConversion = openbabel.OBConversion()

    if not check_filetype(obConversion, filetype):
        die('Input filetype not supported, check http://openbabel.org/docs/2.3.0/FileFormats/Overview.html for supported formats')

    obConversion.SetInAndOutFormats(filetype, "pdb")

    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, options.filename)

    print('Opened molecule with {} atoms and a total mass of {}'.format(mol.NumAtoms(), mol.GetExactMass()))

    return Particles(mol, options)

def main():
    o = handle_arguments() # Getting options
    o = handle_constant_properties(o) # Append constants

    print('Opening molecule')
    particles = handle_molecule_from_file(o)

    # Create a suitable filename
    cleaned_filename = o.filename.split('/')[-1].split('.')[0]
    filename = '{}/{}_N={}_dt={}_l={}_T={}_B={}_{}_{}_Particle.h5' \
        .format(o.data_dir, cleaned_filename, o.N_simulation, o.dt, o.l, o.T, o.B[0], o.B[1], o.B[2])

    # Simulated annealing attempts to find a ground state for the system by gradually lowering the temperature.
    if o.anneal:
        print('Annealing')
        particles = anneal_particles(o, particles)

    # Begin the main simulation phase
    print('Starting simulation')
    results = simulation_iterator(o, particles)

    # Save the raw results
    print ('Done simulating, saving results')
    results.to_hdf(filename, 'df')

    # Plot if needed.
    print('Saved to {}'.format(filename))
    if o.should_plot:
        print('Plotting results')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(results['pos_x'], results['pos_y'], results['pos_z'])

        #plt.xlim(-1, 1)
        #plt.ylim(-1, 1)
        #plt.zlim(-1, 1)

        plt.show()

    return results

if __name__ == '__main__':
    main()
