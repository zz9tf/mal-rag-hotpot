import os
import yaml
from dotenv import load_dotenv

def load_configs():
    """return config, prefix_config"""
    script_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_path, 'config.yml')
    with open(config_path, 'r') as config:
        config = yaml.safe_load(config)
    load_dotenv(dotenv_path=os.path.join(script_path, '.env'))
    return config