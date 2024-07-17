# Harmonic 1-forms on T3
Second repository for summer work (Project 2) at Imperial in collaboration with Hugo Rojbins, under supervision of Daniel Platt and Daattavya Argarwal.

## Packages and Requirements
Install packages in requirements.txt. Tested with Python 3.9.6.

Install Python and Tensorflow.

## T3-PINNS
This project involves developing a PINN to solve the laplace problem Δ𝜂 = −𝑑∗ (𝑑𝑥𝑖 ) for a 0-form and check if it has zeros on some metric, g, we must find.


### Project 2
Does there exist a metric on the 3-dimensional torus 𝑇3 such that every harmonic 1-form has a
zero?

Initial guess: yes. Goal of project is to produce numerical evidence for Yes/No. Define some metric on 𝑇3, compute the harmonic 1-forms and check if they have zeros. 

Initial work: Developed simple model for solving an ODE f''=sin(x), case shown in ode_simple_solver.py, used ideas displayed in [this paper (1)](https://arxiv.org/abs/1711.10561).

Developed periodic PINN on 𝑇3 to solve the laplacian of a 1 form [IN PROGRESS]

## References:
(1) [Raissi et al., 2017, *Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations*](https://arxiv.org/abs/1711.10561)
