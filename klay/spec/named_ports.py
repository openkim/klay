# klay/spec/named_ports.py
def resolve_references(spec: dict):
    """Turn 'layer.port' strings into ('layer', 'port') tuples."""
    for lname, ldata in spec["model_layers"].items():
        inputs = ldata.get("inputs", {})
        new_map = {}
        for port_name, ref in inputs.items():
            layer, *rest = ref.split(".", 1)
            src_port = rest[0] if rest else None
            new_map[port_name] = (layer, src_port)
        ldata["inputs"] = new_map
