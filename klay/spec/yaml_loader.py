from pathlib import Path
from typing import Union

import yaml


def load_yaml(path: Union[str, Path]) -> dict:
    """
    Load a YAML file and return its content as a dictionary.
    :param path: Path to the YAML file.
    :return: Content of the YAML file as a dictionary.
    """
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping")
    if "layers" not in data or not isinstance(data["layers"], list):
        raise ValueError("Expect key 'layers' : [ ... ]")
    return data
