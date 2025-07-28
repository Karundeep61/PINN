Physics-Informed Neural Networks for Fluid Dynamics
This repository contains Python scripts that leverage Physics-Informed Neural Networks (PINNs) to solve complex fluid dynamics problems governed by the Navier-Stokes equations. Instead of relying on traditional numerical solvers, this approach uses the power of deep learning and automatic differentiation in PyTorch to find solutions that are inherently constrained by the laws of physics.

What is a PINN?
A Physics-Informed Neural Network is a type of neural network that is trained to solve supervised learning tasks while respecting any given law of physics described by a general nonlinear partial differential equation (PDE).

The core idea is to embed the PDE directly into the loss function of the neural network. The total loss is a combination of:

Data Loss: The standard loss from labeled data points (e.g., initial and boundary conditions).

Physics Loss: The residual of the governing PDE. The network is penalized if its predictions do not satisfy the physical laws.

By minimizing this combined loss, the network learns a function that not only fits the data but also obeys the underlying physics of the system.

Projects Included
This repository contains two primary examples:

Lid-Driven Cavity: A classic benchmark problem in computational fluid dynamics (CFD) used to validate solvers.

Blood Flow in a Stenosed Artery: A biomedical simulation modeling pulsatile blood flow through a partially blocked artery.

Setup and Installation
This project requires a specific Python environment to ensure compatibility with the necessary CUDA-enabled PyTorch libraries.

Prerequisites
An NVIDIA GPU with CUDA support.

Anaconda or Miniconda installed.

1. Create a Conda Environment
First, create a dedicated Conda environment using a compatible Python version (e.g., 3.11).

# Create an environment named 'pinn_env' with Python 3.11
conda create -n pinn_env python=3.11

# Activate the new environment
conda activate pinn_env

2. Install Dependencies
With the pinn_env activated, install the required libraries.

A. PyTorch with CUDA Support:
Install the appropriate PyTorch version for your CUDA toolkit. The command below is for CUDA 12.1. For other versions or for the latest nightly builds (required for very new GPUs), please refer to the Official PyTorch Website.

# For stable PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

B. Matplotlib:

pip install matplotlib numpy

How to Run the Simulations
Project 1: Lid-Driven Cavity
This simulation models the flow inside a square cavity where the top lid moves at a constant velocity, creating a vortex.

To Run:

python LidDrivenCavity.py

Expected Output:
The script will first show the training progress in the terminal. After training, it will display two plots:

A log-scale plot of the training loss over epochs.

A streamplot visualizing the velocity field, clearly showing the primary vortex.

(Placeholder for your Lid-Driven Cavity plot)

Project 2: Blood Flow in a Stenosed Artery
This simulation models pulsatile blood flow through an artery with a partial blockage (stenosis), a problem of significant medical importance.

To Run:

python StenosedArtery.py

Expected Output:
The script will train the network to solve for the pulsatile flow. After training, it will display:

The training loss curve.

A streamplot of the blood velocity field at a specific point in the cardiac cycle (peak systole), visualizing the high-speed jet passing through the stenosis and the recirculation zones that form downstream.

(Placeholder for your Stenosed Artery plot)

Future Work
Extend the models to 3D simulations.

Incorporate real-world geometry from medical imaging data (e.g., .stl or .step files).

Implement turbulence models for high Reynolds number flows.

Perform inverse modeling to determine physical parameters (like blood viscosity) from velocity data.

Acknowledgements
This work was developed with the assistance of Google's Gemini. The project is inspired by the foundational papers on Physics-Informed Neural Networks by Mazyar Raissi, Paris Perdikaris, and George Em Karniadakis.
