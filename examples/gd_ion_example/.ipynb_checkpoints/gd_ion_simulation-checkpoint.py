from SimulationBaseClass import BaseSimulation
import numpy as np


class GdIonSimulation(BaseSimulation):
    def __init__(self):
        super().__init__() # Run init of superclass, IMPORTANT!

        # Start by configuring the simulation
        self.options['simulation_name'] = 'GdIonSimulation'
        self.options['input_file'] = 'examples/gd_ion_example/gd_ion.pdb'
        self.options['data_file'] = 'examples/gd_ion_example/data.h5' # Saves the data file
        self.options['transform_file'] = 'examples/gd_ion_example/transformed_data.h5' # Saves the transforms file
        self.options['spin'] = 7/2
        self.options['l'] = 0
        self.options['dt'] = 1e-14
        self.options['J'] = 0 # We only have one atom
        self.options['l'] = 0 # No dampening
        self.options['T'] = 0 # Zero temperature
        self.options['B'] = np.array([0, 0, 1]) # 1 tesla in z direction


if __name__ == '__main__':
    # Prep the simulation
    sim = GdIonSimulation()
    sim.load_particles()
    sim.run_anneal(1.6e3)

    # Run the simulation
    sim.run_simulation(1e5)

    # Run the transformations on a range of scattering vectors
    for q_size in np.arange(0., 0.1, 0.1):
        q = q_size * np.array([0, 0, 1])
        sim.run_transformations(q)

    # Plot results
    # Spins
    sim.plot_spins_xy('examples/gd_ion_example/spin_xy.png')
    sim.plot_spins_xyz('examples/gd_ion_example/spin_xyz.png')

    # Energies
    sim.plot_energies('examples/gd_ion_example/energies_{}.png', [0, 0.2])

    # Frequencies
    sim.plot_frequencies('examples/gd_ion_example/frequencies_{}.png', [0, 1e11])

    # Scattering cross section
    sim.plot_scattering_cross_section('examples/gd_ion_example/cross_section_{}.png')

    # Close the simulation
    sim.close()

    print('expected energy: {} meV'.format(- sim.constants['g'] * sim.constants['mu_b_meV']))
