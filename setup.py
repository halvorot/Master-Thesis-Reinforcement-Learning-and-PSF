
from setuptools import setup

setup(
    name='gym-rl-mpc',
    version='0.0.1',
    python_requires='>=3.7, <3.8',
    url='https://github.com/halvorot/gym-rl-mpc',
    install_requires=[
        'cvxpy',
        'tensorflow',
        'numpy==1.19.3',
        'gym',
        'stable-baselines3[extra]',
        'matplotlib',
        'casadi',
        'mosek',
    ]  # And any other dependencies it needs
)
