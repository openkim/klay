import torch
from e3nn.o3 import Irreps

import yaml
import sys

if len(sys.argv)!=2:
    raise ValueError("Should provide only one argument, which is the yaml config file")

yaml_file = sys.argv[1]
print(f"Yaml file: {yaml_file}")

with open(yaml_file, "r") as stream:
    config = yaml.safe_load(stream)
print(config)
n_elems = len(config["elements"])
