import os
import sys
from pprint import pprint
from dotmap import DotMap
import json 

def makedirs(dir_list):
    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)

def process_config(config_path, override_dotmap=None):
    config_json = load_json(config_path)
    return _process_config(config_json, override_dotmap=override_dotmap)

def load_json(f_path):
    with open(f_path, 'r') as f:
        return json.load(f)

def save_json(obj, f_path):
    with open(f_path, 'w') as f:
        json.dump(obj, f, ensure_ascii=False)

def _process_config(config_json, override_dotmap=None):
    """
    Processes config file
    """
    config = DotMap(config_json)
    if override_dotmap is not None:
        config.update(override_dotmap)

    print("Loaded configuration: ")
    pprint(config)

    print()
    print(" *************************************** ")
    print("      Running experiment {}".format(config.exp_name))
    print(" *************************************** ")
    print()

    # NOTE: current setup overwrite every time...

    # Uncomment me if you wish to not overwrite
    # timestamp = strftime('%Y-%m-%d--%H_%M_%S', localtime())
    exp_dir = os.path.join(config.exp_base, "experiments", config.exp_name)
    config.exp_dir = exp_dir

    # create some important directories to be used for the experiment.
    config.checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    config.out_dir = os.path.join(exp_dir, "out/")
    config.log_dir = os.path.join(exp_dir, "logs/")

    # will not create if already existing
    makedirs([config.exp_dir, config.checkpoint_dir, config.out_dir, config.log_dir])

    # save config to experiment dir
    config_out = os.path.join(exp_dir, 'config.json')
    save_json(config.toDict(), config_out)

    return config