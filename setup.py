
from setuptools import setup

setup(
    name='gym-rl-mpc',
    version='0.0.1',
    python_requires='>=3.7, <3.9',
    url='https://github.com/halvorot/gym-rl-mpc',
    install_requires=[
        'gym',
        'stable-baselines3[extra]',
        'matplotlib',
        'termcolor',
        'casadi',
    ]  # And any other dependencies it needs
)
