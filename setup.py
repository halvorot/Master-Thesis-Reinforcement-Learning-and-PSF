
from setuptools import setup

setup(
    name='gym-turbine',
    version='0.0.1',
    python_requires='>=3.7, <3.9',
    url='https://github.com/halvorot/gym-turbine',
    install_requires=[
        'ray[rllib]',
        'gym',
        'matplotlib'
    ]  # And any other dependencies it needs
)
