import sys
from setuptools import setup, find_packages


# Hack to prevent accidental upload to public PyPi server.
# TODO: Permanent solution is for SoarTech to create a private PyPi server and configure packages to deploy there.
def forbid_publish():
    argv = sys.argv
    blacklist = ['register', 'upload']

    for command in blacklist:
        if command in argv:
            values = {'command': command}
            print('Command "%(command)s" has been blacklisted, exiting...' %
                  values)
            sys.exit(2)

forbid_publish()

setup(
    name='deepagent',
    version='1.0.0',
    packages=find_packages(exclude=["*.tests.*"]),
    private_repository='https://pypi.soartech.com',
    description='The DeepAgent PhaseI codebase.',
    entry_points={
        'console_scripts': [
            'deepagent=deepagent.experiments.experiment_runner:main',
            'deepagent_eval=deepagent.experiments.network_evaluator:main'
        ]
    },

    install_requires=[
        'gym==0.12.1',
        'h5py==3.1.0',
        'networkx==2.5.1',
        'perlin-noise==1.7',
        'rtree==0.9.7',
        'trueskill>=0.4.5',
        'scikit-image==0.18.1',
        'sknw==0.14',
        'netifaces==0.11.0',
        'pygame==2.0.0',
        'matplotlib==3.2.2',
        'pysc2==3.0.0',
        'scikit-learn==0.24.1',
        'psutil==5.7.3',
        'pyzmq==19.0.1',
        'tensorflow==2.6.0',
        'keras==2.6',
        'Cython',
    ],

    extras_require={
        "linux": ['getch==1.0',
                  ],
        "ride_env": ['transitions', 'graphviz', 'liac-arff', 'wandb', 'pandas', 'torch']
    },
)

