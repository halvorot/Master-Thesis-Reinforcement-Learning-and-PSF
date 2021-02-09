# gym-turbine
A repository for TTK4900 Master thesis at NTNU. Project is stabilizing floating wind turbines using Deep Reinforcement Learning

## required software
 - OpenFAST
  - Visual Studio (needed for OpenFAST)
  - Intel oneAPI Base toolkit (for fortran compiler needed for OpenFAST) 
  - Intel oneAPI HPC toolkit (for fortran compiler needed for OpenFAST)
  - Follow the guide at https://openfast.readthedocs.io/en/master/source/install/install_vs_windows.html

## Installation
Install the package by running
```
pip install -e ./gym-turbine/
```

## Running the program

To train an agent run:
```
python train.py --timesteps <number of timesteps to train>
```

To simulate the system with an agent run:
```
python run.py --agent <path to agent .pkl or .zip file>
```

To show an animation of the simulation run:
```
python animate.py --data <path to .csv data file> --time <Max simulation time (seconds)>
```
or to animate a simulation of an agent directly run:
```
python run.py --agent <path to agent .pkl or .zip file> --time <Max simulation time (seconds)>
```
