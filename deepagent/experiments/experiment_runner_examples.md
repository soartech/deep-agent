# experiment_runner.py examples

Setup the ips and battle mode via params.NetworkConfig.

## Training Example

**python -m deepagent.experiments.experiment_runner -p params4 -et train**

or from the params directory:

**nohup python -u ../../experiment_runner.py -p params4 -et train &> train_1.out &**

This will start training params4 against the built-in AI. The training weights and logs will be saved in deepagent/experiments/params/params4/<n>/
where <n> is the next highest integer.

## Testing Example

**python -m deepagent.experiments.experiment_runner -p params4 -et test -w 4**

This will start testing params4 against the built-in AI, using the weights found in deepagent/experiments/params/params4/ 


## Andy's commands

```
from .../params/ag_population_ppo_sc2
python -u ../../experiment_runner.py -p ag_population_ppo_sc2 -et train
python -u ../../experiment_runner.py -p ag_population_ppo_sc2 -et train &> train_1.out &
nohup python -u ../../experiment_runner.py -p ag_population_ppo_sc2 -et train &> train_1.out &
python -u ../../experiment_runner.py -p ag_dense_ppo_python_sc2 -et test -w 15
```
