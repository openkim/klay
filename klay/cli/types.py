import click
from rich.console import Console
from rich.table import Table

from ..core.categories import ModuleCategory


@click.command("types")
def list_types():
    """Show all layer categories (ModuleCategory enum)."""
    table = Table(title="Available Layer Categories")
    table.add_column("Category", style="bold cyan")
    table.add_column("Description")

    for cat in ModuleCategory:
        table.add_row(cat.name.lower(), getattr(cat, "__doc__", "") or "--")

    Console().print(table)
