# from generous help of chatgpt!
import pathlib
import sys
from html import escape as _esc

import click
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot
from rich.console import Console
from rich.text import Text

from ..builder.dag import build_dag
from ..io.load_config import load_config

console = Console()


def _find_dangling_layers(g: nx.DiGraph) -> set[str]:
    sinks = {n for n, d in g.nodes(data=True) if d["kind"] == "output"}
    layers = {n for n, d in g.nodes(data=True) if d["kind"] == "layer"}
    return {l for l in layers if not any(nx.has_path(g, l, s) for s in sinks)}


def _collect_soft_warnings(g: nx.DiGraph) -> list[str]:
    warn = []

    # 1) alias -> alias (double-hop)
    for n, d in g.nodes(data=True):
        if d.get("kind") == "alias" and g.nodes[d["target"]].get("kind") == "alias":
            warn.append(f"Alias '{n}' targets another alias '{d['target']}'.")

    # 2) layer without explicit inputs
    for n, d in g.nodes(data=True):
        if d.get("kind") == "layer" and g.in_degree(n) == 0:
            warn.append(f"Layer '{n}' has no declared inputs.")

    # 3) layer with outputs mapping but none of those aliases are used
    for n, d in g.nodes(data=True):
        if d.get("kind") == "layer":
            out_aliases = {
                a
                for a, ad in g.nodes(data=True)
                if ad.get("kind") == "alias_out" and ad["parent"] == n
            }
            used = any(a in out_aliases for a in _ancestors_used(g, out_aliases))
            if out_aliases and not used:
                warn.append(f"Outputs of layer '{n}' are never consumed.")

    return warn


def _ancestors_used(g: nx.DiGraph, aliases: set[str]) -> set[str]:
    used = set()
    for a in aliases:
        used |= nx.descendants(g, a)
    return used


def _render_graph(g: nx.DiGraph, path: pathlib.Path, fmt: str):
    """Colour-code nodes and dump via Graphviz."""
    pd = to_pydot(g)

    for n, data in g.nodes(data=True):
        # pydot may quote / mangle names; fall back gracefully
        try:
            pd_node = pd.get_node(str(n))[0]
        except IndexError:
            # create a fresh pydot.Node so we can still style it
            from pydot import Node as _PDNode

            pd_node = _PDNode(str(n))
            pd.add_node(pd_node)

        kind = data["kind"]
        if kind == "input":
            _style(pd_node, "box", "#A1D99B")
        elif kind == "output":
            _style(pd_node, "box", "#FDD0A2")
        elif kind == "alias":
            _style(pd_node, "ellipse", "#CBC9E2")
        elif kind == "alias_out":
            _style(pd_node, "note", "#D9D9D9")
        else:  # layer
            _style(pd_node, "ellipse", "#9ECAE1")

        # nicer label (escape {} etc.)
        pd_node.set_label(_esc(str(n)))

    pd.write(path.as_posix(), format=fmt)


def _style(node, shape, color):
    node.set_shape(shape)
    node.set_style("filled")
    node.set_fillcolor(color)


@click.command("validate")
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--allow-dangling", is_flag=True, help="Treat dangling layers as warnings instead of errors."
)
@click.option("-v", "--visualize", is_flag=True, help="Render DAG image next to CONFIG_PATH.")
@click.option(
    "--fmt",
    default="png",
    type=click.Choice(["png", "svg", "pdf", "jpeg"]),
    help="Image format when --visualize is used.",
)
def validate_cfg(config_path, allow_dangling, visualize, fmt):
    """
    Validate CONFIG_PATH. Checks:

      - cycles / missing sources (fatal)
      - dangling layers (fatal unless --allow-dangling)
      - alias->alias, unused outputs, etc. (warnings)
      - optional: save diagram (--visualize)
    """
    try:
        cfg = load_config(config_path)
        dag = build_dag(cfg)
    except Exception as exc:
        console.print(Text("Invalid:", style="bold red"), str(exc))
        sys.exit(1)

    # ---- hard error: dangling ---------------------------------------------
    dangling = _find_dangling_layers(dag)
    if dangling and not allow_dangling:
        console.print(Text("Dangling layers:", style="bold red"), sorted(dangling))
        sys.exit(2)

    # ---- soft warnings ----------------------------------------------------
    warns = _collect_soft_warnings(dag)
    if dangling and allow_dangling:
        warns.append(f"Dangling layers: {sorted(dangling)}")
    for w in warns:
        console.print(Text("Warning:", style="yellow"), w)

    # ---- visualize --------------------------------------------------------
    if visualize:
        img_path = pathlib.Path(config_path).with_suffix(f".{fmt}")
        try:
            _render_graph(dag, img_path, fmt)
            console.print(Text("Graph written ->", style="cyan"), img_path)
        except Exception as exc:
            console.print(Text("Could not render graph:", style="yellow"), str(exc))

    # ---- summary ----------------------------------------------------------
    console.print(Text("Config is valid!", style="bold green"))
