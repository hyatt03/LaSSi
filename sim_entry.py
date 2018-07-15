#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cli_helper import handle_arguments, get_data_dir
from SimulationBaseClass import BaseSimulation
import numpy as np


def main():
    o = handle_arguments() # Getting options

    # Initialize a simulation
    sim = BaseSimulation()

    # Configure the simulation
    sim.options['simulation_name'] = 'CommandlineSimulation'
    sim.options['input_file'] = o.filename
    sim.options['data_file'] = o.data_dir + '/data.h5'
    sim.options['transform_file'] = o.data_dir + '/transforms.h5'
    sim.options['spin'] = o.spin
    sim.options['l'] = o.l
    sim.options['T'] = o.T
    sim.options['J'] = o.J
    sim.options['B'] = o.B
    sim.options['dt'] = o.dt

    # Load the molecule
    print('Opening molecule')
    sim.load_particles()

    # Create a suitable data directory
    data_dir = get_data_dir(o)

    if o.anneal:
        print('annealing')
        sim.run_anneal(o.anneal)

    # Run the simulation
    print('simulating')
    sim.run_simulation(o.N_simulation)

    # Run the transformations
    if isinstance(o.q, np.ndarray):
        print('transforming')
        sim.run_transformations(o.q)

    # Plot the spins
    if o.should_plot_spins:
        print('plotting spins')
        sim.plot_spins_xy(o.data_dir + '/spin_xy.png')
        sim.plot_spins_xyz(o.data_dir + '/spin_xyz.png')

    # Plot the energies
    if o.should_plot_energy:
        print('plotting energy')
        sim.plot_energies(o.data_dir + '/energies_{}.png', [0, 0.2])

    # Close the simulation
    sim.close()


if __name__ == '__main__':
    main()
