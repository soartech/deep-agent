# DeepAgent

This repository is used for developing, training, and testing deep reinforcement learning algorithms in multiple
different simulation environments.

## Installation
### Windows-specific notes:
- must run PowerShell as Administrator in order to run venv
  - activate the venv by calling ./<venv_path>/Scripts/activate (which should call activate.ps1)
- there seems to be a conflict with the version of protobuf that gets installed by default (3.6.1). Manually have to install protobuf 3.8.0
- must install "visual studio 2015 2019 x86 redistributable" for tensorflow

### Python 3.8.2 notes
- Project appears to work fine when using python 3.8.2

Create a pip virtual environment and install editable version of deepagent.
Tensorflow and tensorflow-gpu must be completely uninstalled prior to installing the correct version of tensorflow-gpu to prevent conflicts.

OPTIONAL flags for deep-agent install: ride_env, linux

```
python -m venv venv
# LINUX-ONLY
source venv/bin/activate
# WINDOWS-POWERSHELL ONLY
.\venv\Scripts\Activate.ps1

# basic python env updates
# required pip version > 19.0
python -m pip install --upgrade pip 
# used version 41.0.1
python -m pip install --upgrade setuptools

# project installs
git clone https://hq-git.soartech.com/deepagent/deep-agent.git
git clone https://github.com/openai/baselines.git
git checkout c575285
# change tf.set_random_seed(myseed) to tf.random.set_seed(myseed)
vim common/misc_util.py

python -m pip install -e deep-agent
python -m pip install -e baselines

# LINUX ONLY: Add the CuDNN install to your path
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

## Environments Supported
- OpenAI Atari
- RACER

## Running

```
# training
python -m deepagent.experiments.experiment_runner -p nature_ppo_racer -et train
# testing
python -m deepagent.experiments.experiment_runner -p nature_ppo_racer -et test
```