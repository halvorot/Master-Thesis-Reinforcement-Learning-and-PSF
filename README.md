# gym-turbine
A repository for TTK4900 Master thesis at NTNU. Project is stabilizing a pendulum using Reinforcement Learning and MPC

## Prerequisites
 - Python 3.7

## Installation
Run
```
pip install -e ./gym-rl-mpc
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
