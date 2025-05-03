# klay/spec/interpolate.py
import re

_TOKEN = re.compile(r"\$\{global\.([A-Za-z0-9_]+)\}")


# TODO OmegaConf
def interpolate_globals(raw: dict) -> dict:
    g = raw.get("global", {})
    if not g:
        return raw

    for layer in raw["layers"]:
        kwargs = layer.get("kwargs", {})
        for key, val in list(kwargs.items()):
            if not isinstance(val, str):
                continue

            m = _TOKEN.fullmatch(val)
            if m:
                kwargs[key] = g[m.group(1)]
                continue

            kwargs[key] = _TOKEN.sub(lambda m: str(g[m.group(1)]), val)

    return raw
