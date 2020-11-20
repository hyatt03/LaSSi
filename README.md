# LaSSi

![LaSSi Logo](https://i.imgur.com/DVhJ5W4.jpg "LaSSi Logo - Image is free of rights")

LaSSi is a classical spin simulations package that uses Langevin dynamics to simulate at finite temperature. 
It is class based, pure python, easy to use, and easy to install.

## How does it work

LaSSi works by integrating a classical approximation of the spin movements over time, a system can be selected by opening a file containing crystallographic information such as the unit cell type and the positions of the individual atom. Using this information we compute the nearest neighbours and from that evolve the motion through time. The time dependent spacial positions of the atoms can then be fourier transformed to give a proportionality to the neutron scattering cross section, thereby making it possible to model systems with large effective spins. Temperature is simulated through Langevin dynamics giving results that would otherwise be difficult to obtain through analytical computations.

## Examples

Examples can be found in the examples folder.
