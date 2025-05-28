from pathlib import Path

import pytest
from e3nn.util import jit

from klay.builder import build_model
from klay.io import load_config


def test_model_nequip():
    examples_dir = Path(__file__).parent.parent / "example"
    cfg = load_config(examples_dir / "new_model.yaml")
    model = build_model(cfg)
    model = jit.script(model)

    assert model is not None


def test_mace_model():
    examples_dir = Path(__file__).parent.parent / "example"
    cfg = load_config(examples_dir / "mace_model.yaml")
    model = build_model(cfg)
    model = jit.script(model)

    assert model is not None


def test_alias_model():
    examples_dir = Path(__file__).parent.parent / "example"
    cfg = load_config(examples_dir / "arbitrary_and_alias.yaml")
    model = build_model(cfg)

    print(model)

    assert model is not None
