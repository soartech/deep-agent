import importlib
import inspect
import os

from deepagent.experiments.params import params


def get_custom_params_source(name='params1'):
    my_params = importlib.import_module('.' + name + '.' + name, package='deepagent.experiments.params')
    return inspect.getsource(my_params)

def get_dev_params_source(modulename='params1', sourcename='dev_experiment_param_override'):
    my_params = importlib.import_module('.' + modulename + '.' + sourcename, package='deepagent.experiments.params')
    return inspect.getsource(my_params)


def load_params(name='params1'):
    # The params modules live in .<name>.<name> relative to this module
    my_params = importlib.import_module('.' + name + '.' + name, package='deepagent.experiments.params')
    params.ModuleParams.name = name

    # Dynamically updating the weights_dir to point to the parent directory of the params
    # Module that is overwriting the params stub module
    params.ModuleParams.params_dir = os.path.abspath(os.path.dirname(my_params.__file__))


if __name__ == '__main__':
    load_params('params1')
    print(params.ModuleParams.params_dir)


def write_params():
    with open(os.path.join(params.ModuleParams.weights_dir, params.ModuleParams.name + '.txt'),
              'w') as custom_params_file:
        custom_params_file.write(get_custom_params_source(params.ModuleParams.name))

    with open(os.path.join(params.ModuleParams.weights_dir, 'params.txt'), 'w') as default_params:
        default_params.write(inspect.getsource(params))

    with open(os.path.join(params.ModuleParams.weights_dir, "file_overrides" + '.txt'),
              'w') as dev_params_file:
        dev_params_file.write(get_dev_params_source(params.ModuleParams.name))


def write_commandline_params(argstr="", log="commandline_overrides"):
    with open(os.path.join(params.ModuleParams.weights_dir, log + '.txt'),
              'w') as cmd_params_file:
        cmd_params_file.write(argstr)

def write_params_state(name="params_final_state"):
    with open(os.path.join(params.ModuleParams.weights_dir, name + '.txt'),
              'w') as state_params_file:
        state_params_file.write(params.get_param_state())
