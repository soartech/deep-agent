# DeepAgent

This repository is used for developing, training, and testing deep reinforcement learning algorithms in multiple
different simulation environments.

## Installation
```
# setup python 3.8 venv in pycharm

# project installs
git clone https://github.com/soartech/deep-agent.git
git clone https://github.com/soartech/baselines.git

pip install -e deep-agent
pip install -e baselines
```

## Environments Supported
- OpenAI Atari
- RACER

## Running

```
# Training
python -m deepagent.experiments.experiment_runner -p nature_ppo_racer -et train
# Testing
python -m deepagent.experiments.experiment_runner -p nature_ppo_racer -et test -w <numbered weights directory, e.g. 3>
```
