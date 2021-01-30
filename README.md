# LaSSi

![LaSSi Logo](https://i.imgur.com/DVhJ5W4.jpg "LaSSi Logo - Image is free of rights")

LaSSi is a classical spin simulations package that uses Langevin dynamics to simulate at finite temperature. 
It is class based, pure python, easy to use, and easy to install.

## How does it work

LaSSi works by integrating a classical approximation of the spin movements over time, a system can be selected by opening a file containing crystallographic information such as the unit cell type and the positions of the individual atom. Using this information we compute the nearest neighbours and from that evolve the motion through time. The time dependent spacial positions of the atoms can then be fourier transformed to give a proportionality to the neutron scattering cross section, thereby making it possible to model systems with large effective spins. Temperature is simulated through Langevin dynamics giving results that would otherwise be difficult to obtain through analytical computations.

## Examples

Examples can be found in the examples folder.

## Status of the project

Currently LaSSi is very much a work in progress, we're currently working on scientific validation of the methods applied, as well as correcting any mistakes that may have occurred along the way. We aim to publish a PyPI package of the entire projekt so anybody can create and run simulations, but until then just check it out and have a look at the examples.
