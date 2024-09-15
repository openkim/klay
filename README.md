![](./KlayLogo.png)
# KLay - KLIFF Layers, trainable and pre-trained layers for MLIPs
[![Documentation Status](https://readthedocs.org/projects/klay/badge/?version=latest)](https://klay.readthedocs.io/en/latest/?badge=latest)
 
KLIFF compatible ML layers you can mould to your needs. Usable for general neural network architectures as well. Works out of the box with KLIFF.


The idea of this package is to generate different kind of layers from simple yaml like input that the user can then assemble on their own for research and experimentation.
Lot of different packages provide complete opaque models that are easy to use but hard to modify. 
This package is supposed to take a complementary approach where the user can easily modify the layers to their needs.

Currently only NequIP layers are supported. More layers will be added in the future.

## Installation
```bash
pip install klay
```

## Usage
```python
from klay import get_model_layers_from_yaml

layers = get_model_layers_from_yaml("path/to/yaml")
```

Example yaml file:
```yaml
model:
    - elem_embedding:
        embedding_type: one_hot
        n_elems: 1

    - edge_embedding:
        lmax: 1
        normalize: True
        normalization: component
        parity: True
```

## Documentation
The documentation can be found [here](https://klay.readthedocs.io/en/latest/)

## Future Work
- [ ] MACE Tensor Product Layer
- [ ] EGNN Layers
- [ ] Pretrained layers like M3GNET