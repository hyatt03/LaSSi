#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('agg')

import json
import pandas as pd
import numpy as np
from cli_helper import handle_arguments, handle_constant_properties, get_data_dir
from Particles import handle_molecule_from_file
from simulation_iterator import simulation_iterator, plot_spins
from annealing import parrallel_anneal
from fourier import fourier, plot_fourier, plot_energy_spectrum, calculate_scattering_intensity, parallel_compute_scattering_intensity
import matplotlib.pyplot as plt


def main():
    o = handle_arguments() # Getting options
    o = handle_constant_properties(o) # Append constants

    print('Opening molecule')
    particles = handle_molecule_from_file(o)

    # Create a suitable data directory
    data_dir = get_data_dir(o)

    # Save the options of this run
    with open('{}/parameters.json'.format(data_dir), 'w') as params:
        options_dict = vars(o)
        for key, value in options_dict.items():
            if type(value) == type(np.array([])):
                options_dict[key] = value.tolist()

        params.writelines(json.dumps(options_dict))

    # Simulated annealing attempts to find a ground state for the system by gradually lowering the temperature.
    if o.anneal:
        print('Annealing')
        particles = parrallel_anneal(o, particles, 8)

    if o.datafile:
        print('Loading datafile')
        results = pd.read_hdf(o.datafile)
        print('Loaded {} rows'.format(len(results)))
    else:
        # Begin the main simulation phase
        print('Starting simulation')
        results = simulation_iterator(o, particles)

        # Save the raw results
        print ('Done simulating')
        filename = '{}/datafile.h5'.format(data_dir)
        results.to_hdf(filename, 'df')

        print('Saved data to {}'.format(filename))

    # Plot if needed.
    if o.should_plot_spins:
        print('Plotting spins')
        plot_spins(results, '{}/spin_plot'.format(data_dir))

    # Runs a fourier transform
    if o.fourier:
        print('Transforming results')
        # Plot the transformed variables
        if o.should_plot_energy:
            print('Plotting transformed results')
            I_aa_temp, energy, frequency = fourier(o, results, particles)
            plot_fourier(o, '{}/energy_q0.png'.format(data_dir), I_aa_temp, frequency, energy)

        if o.should_plot_neutron:
            print('Plotting energy spectrum')
            qs, I_aa_temps, energies, frequencies = parallel_compute_scattering_intensity(o, results, particles)

            fig, ax = plt.subplots()

            energy = energies[0]
            for temp in I_aa_temps:
                y_data = (temp[0][0:len(energy)] + temp[1][0:len(energy)] + temp[2][0:len(energy)]) / 3
                ax.plot(energy, y_data)

            plt.xlim(0, 0.6)
            plt.xlabel('Energy [meV]')
            plt.ylabel('Intensity [A.U.]')

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            plt.legend(['q = {}'.format(q) for q in qs], loc='center left', bbox_to_anchor=(1, 0.5))

            plt.savefig('{}/multi_q_spektrum.png'.format(data_dir), bbox_inches='tight', dpi=300)

            # plot_energy_spectrum(o, '{}/energy_spectrum.png'.format(data_dir), qs, I_aa_temps, energies, frequencies)

    return results


if __name__ == '__main__':
    main()
