from SimulationBaseClass import BaseSimulation
from vectorized_simulation_iterator import vectorized_simulation_iterator


class VectorizedBaseSimulation(BaseSimulation):
    def run_simulation(self, iterations):
        # Check if the sim is ready to start
        self.check_sim_can_start()

        # Open the datafile first
        self.open_datafile()

        # Run the sim
        vectorized_simulation_iterator(self.options, self.constants, self.particles, iterations, self.datatables, 0)
