
from setuptools import setup

setup(
    name='gym-turbine',
    version='0.0.1',
    python_requires='>=3.7, <3.9',
    url='https://github.com/halvorot/gym-turbine',
    install_requires=[
        'gym',
        'stable-baselines3[extra]',
        'matplotlib',
        'termcolor',
    ]  # And any other dependencies it needs
)
