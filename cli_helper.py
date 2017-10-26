# -*- coding: utf-8 -*-

import os
import sys
import hashlib
import numpy as np
import json
import math
from utils import dot
from optparse import OptionParser

k_b = 1.38064852e-23


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
    parser.add_option("-J", "--NNIC", dest="J", help="Set the nearest neighbour interaction constant (Kelvin)", metavar="NNIC") # I enheder af kelvin.
    parser.add_option("-N", "--iterations", dest="N_simulation", help="Set the amount of iterations", metavar="ITER")
    parser.add_option("-A", "--anneal", dest="anneal", help="Enable annealing in N steps", metavar="N")
    parser.add_option("-F", "--fourier", dest="fourier", help="Should we fourier transform", metavar="F")
    parser.add_option("-D", "--datafile", dest="datafile", help="Use existing datafile (set path to data)", metavar="D")
    parser.add_option("-q", "--scattering-vector", dest="q", help="Scattering vector used for fourier transformation, comma delimited (-q x,y,z)", metavar="q")
    parser.add_option("--plot-spins", dest="should_plot_spins", help="Do you want to plot the spins?", metavar="P")
    parser.add_option("--plot-energies", dest="should_plot_energy", help="Do you want to plot the energy?", metavar="P")
    parser.add_option("--plot-neutron", dest="should_plot_neutron", help="Do you want to plot the scattering profile?", metavar="P")
    parser.add_option("--parameterfile", dest="parameter_file", help="Use existing parameter file (set path to parameters)", metavar="PARAMS")
    parser.add_option("--dt", dest="dt", help="Set the dt", metavar="DT")

    (options, args) = parser.parse_args()

    # Handle parameter files
    if options.parameter_file:
        print('Loading parameters from {}'.format(options.parameter_file))

        # Read from the json file
        with open(options.parameter_file, 'r') as params:
            o = json.loads(params.readlines()[0])

        for key, value in o.iteritems():
            if key == 'parameter_file':
                value = options.parameter_file # Keep this options for later logic

            if type(value) == type([]):
                value = np.array(value) # Convert arrays back to ndarrays

            setattr(options, str(key), value)

        return options

    if not options.filename and not options.datafile:
        die('An inputfile or datafile is required!')

    # Handle spins (supports fractions)
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
        options.J = float(options.J) * k_b
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

        # Calculate the spherical version of the external B field.
        options.spherical_B = np.array([
            math.sqrt(dot(options.B, options.B)),  # r
            math.atan2(math.sqrt(options.B[0] ** 2.0 + options.B[1] ** 2.0), options.B[2]),  # theta
            math.atan2(options.B[1], options.B[0])  # phi
        ])

    else:
        print('No external B field applied')
        options.B = np.array([0, 0, 0])

    if options.N_simulation:
        options.N_simulation = (2 ** (int(float(options.N_simulation)) - 1)).bit_length()
    else:
        die('You must set the iterations (N_simulation)!')

    if options.dt:
        options.dt = float(options.dt)
    else:
        die('You must set the dt (for example 4.14e-15)!')

    if options.anneal:
        options.anneal = int(float(options.anneal))

    if options.q and ',' in options.q:
        q = options.q.split(',')
        options.q = [0, 0, 0]

        if q[0]:
            options.q[0] = float(q[0])

        if q[1]:
            options.q[1] = float(q[1])

        if q[2]:
            options.q[2] = float(q[2])

        options.q = np.array(options.q)

    return options


def handle_constant_properties(options):
    # Use options provided in file.
    if options.parameter_file:
        return options

    # Set data directory
    # TODO: Optional parameter for data dir.
    options.data_dir = os.path.dirname(os.path.abspath(__file__)) + '/data'

    """
    Physical quantities
    """
    options.k_b = k_b
    options.g = -2.002
    options.mu_b = 9.274009994e-24
    options.hbar = 1.054571800e-34
    options.GHz_to_meV = 0.004135665538536
    options.gamma = (options.g * options.mu_b) / options.hbar

    return options


def get_data_dir(o):
    # Create a foldername to save our results in.
    cleaned_filename = o.filename.split('/')[-1].split('.')[0]
    m = hashlib.md5()
    m.update('{}_{}_{}_{}_{}_{}_{}_{}' \
                    .format(cleaned_filename, o.N_simulation, o.dt, o.l, o.T, o.B[0], o.B[1], o.B[2]))
    directory_name = m.hexdigest()
    data_dir = '{}/{}'.format(o.data_dir, directory_name)

    # Check if data dir exists. If not recursively create it.
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    return data_dir
