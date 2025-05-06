import inspect
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from ..core import get, names
from ..core.categories import ModuleCategory


@click.command("layers")
@click.option(
    "--type",
    "--category",
    "cat",
    metavar="<category>",
    help="Filter by layer category (see `mlip types`).",
)
@click.option(
    "--all", "show_all", is_flag=True, help="Include layers without a 'from_config' method."
)
def list_layers(show_all: bool, cat: Optional[str]):
    table = Table(title="Registered MLIP Layers", show_lines=True)
    table.add_column("Layer", style="bold cyan", no_wrap=True)
    table.add_column("Inputs", style="green")
    table.add_column("Outputs", style="green")
    table.add_column("from_config args", style="yellow")

    if cat:
        try:
            cat_enum = ModuleCategory[cat.upper()]
        except KeyError:
            available = ", ".join(c.name.lower() for c in ModuleCategory)
            raise click.BadParameter(f"Unknown category '{cat}'. " f"Available: {available}")

    for lname in names():
        meta = get(lname)
        if cat and meta.category is not cat_enum:
            continue

        meta = get(lname)
        cls = meta.cls
        has_fc = hasattr(cls, "from_config")
        if not has_fc and not show_all:
            continue

        inputs = ", ".join(meta.inputs)
        if len(meta.outputs) > 1:
            outputs = "\n".join(f"{i}: {name}" for i, name in enumerate(meta.outputs))
        else:
            outputs = ", ".join(meta.outputs)

        if has_fc:
            sig = inspect.signature(cls.from_config)
            params = [p for p in sig.parameters.values() if p.name not in ("cls", "self")]
            arglist = []

            for p in params:
                if p.default is inspect._empty:
                    arglist.append(f"[bright_red]{p.name}[/]")
                else:
                    default_repr = repr(p.default)
                    arglist.append(f"[bright_yellow]{p.name}={default_repr}[/]")
            fc_args = ", ".join(arglist) or "-"
        else:
            fc_args = "-"

        table.add_row(lname, inputs, outputs, fc_args)

    Console().print(table)
