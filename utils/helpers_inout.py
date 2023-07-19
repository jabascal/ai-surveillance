import os

import yaml
from yamlinclude import YamlIncludeConstructor


def load_config(config_file):

    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, 
                                           base_dir=os.path.split(config_file)[0])

    # Import YAML parameters from config/config.yaml
    with open(config_file, 'r') as stream:
        param = yaml.load(stream, yaml.FullLoader)
        print(yaml.dump(param))
    return param

