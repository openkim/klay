import click

from .export import export_model
from .layers import list_layers
from .types import list_types
from .validate import validate_cfg


@click.group()
def cli():
    pass


cli.add_command(list_layers)
cli.add_command(list_types)
cli.add_command(validate_cfg)
cli.add_command(export_model)
