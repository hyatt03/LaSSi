from SimulationBaseClass import BaseSimulation
import numpy as np
from pathlib import Path

basedir = 'examples/interacting_spins_example/'

class GdIonSimulation(BaseSimulation):
    def __init__(self):
        super().__init__() # Run init of superclass, IMPORTANT!

        # Start by configuring the simulation
        self.options['simulation_name'] = 'InteractingSpinsSimulation'
        self.options['input_file'] = basedir + 'two_spins.pdb'
        self.options['data_file'] = basedir + 'data.h5'
        self.options['transform_file'] = basedir + 'transformed_data.h5'
        self.options['spin'] = 7/2
        self.options['l'] = 5e-4
        self.options['dt'] = 1e-16
        self.options['J'] = 1e-6
        self.options['T'] = 0.05
        self.options['B'] = np.array([0, 0, 0.]) # 0.1 tesla in z direction


if __name__ == '__main__':
    # Prep the simulation
    sim = GdIonSimulation()
    sim.load_particles()
    sim.run_anneal(2e5)

    # Run the simulation
    sim.run_simulation(1e5)

    # Run the transformations on a range of scattering vectors
    for q_size in np.arange(0.2, 1.3, 0.1):
        q = q_size * np.array([1, 0, 0])
        sim.run_transformations(q)

    # Plot results
    # Spins
    sim.plot_spins_xy(basedir + 'spin_xy.png'.format(basedir))
    sim.plot_spins_xyz(basedir + 'spin_xyz.png'.format(basedir))

    # Energies
    sim.plot_energies(basedir + 'energies_{}.png', [0, 1])

    # Frequencies
    sim.plot_frequencies(basedir + 'frequencies_{}.png', [0, 1e11])

    # Scattering cross section
    sim.plot_scattering_cross_section(basedir + 'cross_section_{}.png')

    # Close the simulation
    sim.close()

    print('cleaning files')
    Path(sim.options['data_file']).unlink()
    Path(sim.options['transform_file']).unlink()
