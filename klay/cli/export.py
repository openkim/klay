import pathlib

import click
import torch
from e3nn.util import jit
from rich.console import Console

from ..builder.fx_builder import build_fx_model
from ..io.load_config import load_config

console = Console()


@click.command("export")
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "-o", "--out", "out_path", help="Output file (defaults to CONFIG_PATH with .pt/.onnx...)."
)
@click.option(
    "--format",
    type=click.Choice(["pt", "state_dict"]),
    default="pt",
    show_default=True,
    help="Export format. 'pt' = TorchScript, " "'state_dict' = weights only (.pth).",
)
@click.option("--n", type=int, default=4, help="Dummy #atoms for scripting example input.")
def export_model(config_path, out_path, format, n):
    """
    Build MODEL from CONFIG_PATH and export to a serialized file.
    """
    cfg = load_config(config_path)
    model = build_fx_model(cfg).eval()

    if out_path is None:
        suffix = ".pth" if format == "state_dict" else ".pt"
        out_path = pathlib.Path(config_path).with_suffix(suffix)
    out_path = pathlib.Path(out_path)

    if format == "state_dict":
        torch.save(model.state_dict(), out_path)
    else:  # TorchScript
        scripted = jit.script(model)
        scripted.save(out_path.as_posix())

    console.print(f"[bold green] Exported model ->[/] {out_path}")
