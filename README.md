# gym-turbine
A repository for TTK4900 Master thesis at NTNU. Project is stabilizing floating wind turbines using Deep Reinforcement Learning

## Prerequisites
 - ffmpeg (has to be installed with conda, with pip does not work, conda install ffmpeg)
 - slycot (has to be installed with conda, with pip does not work, conda install -c conda-forge slycot)
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

### Training
To train an agent run:
```
python train.py
```
Optional arguments:
- --timesteps <number of timesteps to train the agent>
- --note <Note with additional info about training, gets added to Note.txt>
- --no_reporting <Skip reporting>


### Simulating with a trained agent
To simulate the system with an agent run:
```
python run.py
```
Required arguments:
- --agent <path to agent .zip file>

Optional arguments:
- --time <Max simulation time (seconds)>


### Animating
To show an animation of the simulation run:
```
python animate.py
```
Required arguments:
- --data <path to .csv data file>

Optional arguments:
- --save_video
- --time <Max simulation time (seconds)>

or to animate a simulation of an agent directly run:
```
python animate.py
```
Required arguments:
- --agent <path to agent .zip file>

Optional arguments:
- --save_video
- --time <Max simulation time (seconds)>
