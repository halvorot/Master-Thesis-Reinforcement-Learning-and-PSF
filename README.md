# gym-turbine
A repository for TTK4900 Master thesis at NTNU. Project is stabilizing a pendulum using Reinforcement Learning and MPC

## Prerequisites
 - Matlab 2020b 
 - MPT 3.2.1 w/dependencies 
 - Mosek 9.2.37

## Installation

### Conda env
For improved stability, env is recommend.
Install in order given. 

Setup env and install conda and pip dependecies through:
```
conda env create [-n env_name] -f environment.yml
```

Install the MPT3 toolbox and integrate with Matlab:  
https://www.mpt3.org/Main/Installation

Install mosek (opt: only for quicker solution) and integrate with Matlab/yalmip:
https://www.mosek.com/downloads/


Academic License Mosek
https://www.mosek.com/products/academic-licenses/

Install the Matlab engine for Python:
https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

## Running the program


### Training
To train an agent run:
```
python train.py
```
Optional arguments:
- --agent <path to pretrained agent .zip file to start training from>
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
- --plot


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

### Test PSF dependencies
```
python PSF/py2matlab.py
```


## Based on

https://arxiv.org/pdf/1812.05506.pdf
