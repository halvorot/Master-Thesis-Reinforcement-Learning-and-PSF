# gym-turbine
A repository for TTK4900 Master thesis at NTNU. Project is stabilizing a turbine using Reinforcement Learning and MPC

## Prerequisites
 - Python 3.7
 - Yalmip 20200930
 - Mosek 9.2
 - MATLAB R2011a or later (Tested with 2020b)


## Installation
### Mosek 
Install Mosek with the Default Installers ( Link: https://www.mosek.com/downloads/)

Link Mosek to Matlab (Link : https://docs.mosek.com/9.2/toolbox/install-interface.html)
```
addpath <MSKHOME>/mosek/9.2/toolbox/r2015aom
```
### Yalmip
Yalmip is easily through the txbmananger http://tbxmanager.com/. 
After installation, run the follow command in Matlab:
```
>> tbxmanager install yalmip
```

### Python dependencies and packages

A virtual enviroment is highly recommended. Conda env was used to develop this project.

```
conda create -n <env_name>
```
Run
```
pip install -e ./gym-rl-mpc
```
#### Matlab API 
This must be done in the virtual env, if used. (link: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
```
>>> cd <matlabroot>\extern\engines\python
>>> python setup.py install
```

## Test
Run all the test in gym-rl-mpc folder.
Test Mosek in Matlab:
```
>> mosekdiag
```
Test yalmip in Matlab:
```
>> yalmiptest
```
Test PSF module:
```
>>> python PSF/PSF.p
```

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

To view the tensorboard logs run (in another cmd window):
```
tensorboard --logdir logs
```

NOTE: The environment config is located in gym_rl_mpc/\_\_init\_\_.py

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
To animate a simulation of an agent, run:
```
python animate.py
```
Required arguments:
- --agent <path to agent .zip file>

Optional arguments:
- --save_video
- --time <Max simulation time (seconds)>

Or to show an animation of a simulation run (from file):
```
python animate.py
```
Required arguments:
- --data <path to .csv data file>

Optional arguments:
- --save_video
- --time <Max simulation time (seconds)>

or 




## Based on

https://arxiv.org/pdf/1812.05506.pdf
