
from setuptools import setup

setup(name='gym-turbine',
      version='0.0.1',
      python_requires='>=3.7',
      install_requires=[
          'ray[rllib]',
          'gym'
          ]  # And any other dependencies it needs
      )
