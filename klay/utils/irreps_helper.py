from typing import Any, Dict, List

from e3nn.o3 import Irreps

_PARITY_ALIAS = {
    "e": "e",
    "even": "e",
    "o": "o",
    "odd": "o",
}


def irreps_blocks_to_string(blocks: List[Dict[str, Any]]) -> str:
    """Convert list-of-dict irreps spec -> canonical e3nn string.

    Default parity:
      * absent ->  'e'  if l is even,  'o'  if l is odd

    Example
    -------
    blocks = [
        {"l": 0, "mul": 64},                # 0 => even => 64x0e
        {"l": 1, "mul": 32},                # 1 => odd  => 32x1o
        {"l": 2, "mul": 16, "p": "odd"},    # explicit override => 16x2o
    ]
    >>> irreps_blocks_to_string(blocks)
    '64x0e + 32x1o + 16x2o'
    """
    tokens: list[str] = []
    for i, blk in enumerate(blocks):
        try:
            l = int(blk["l"])
            mul = int(blk["mul"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Block #{i} needs integer 'l' and 'mul' fields.") from exc

        # choose parity
        if "p" in blk:
            try:
                p = _PARITY_ALIAS[blk["p"].lower()]
            except KeyError:
                raise ValueError(f"Block #{i}: parity '{blk['p']}' must be even/odd/e/o")
        else:
            p = "e" if l % 2 == 0 else "o"

        tokens.append(f"{mul}x{l}{p}")

    return " + ".join(tokens)


def irreps_string_to_blocks(s: str):
    """
    Very small inverse of 'irreps_blocks_to_string'.

    Parameters
    ----------
    s : str
        e3nn irreps string like "64x0e + 32x1o + 16x2o"

    Returns
    -------
    List[dict]
        [{'l': 0, 'p': 'e', 'mul': 64}, ...]
    """
    blocks = []
    for token in s.split("+"):
        token = token.strip()
        try:
            mul_part, lp_part = token.split("x")
            mul = int(mul_part)
            l = int(lp_part[:-1])
            p = lp_part[-1].lower()
            if p not in ("e", "o"):
                raise ValueError
        except (ValueError, IndexError):
            raise ValueError(f"Un-parsable token '{token}'") from None

        blocks.append({"l": l, "p": p, "mul": mul})
    return blocks


# def irreps_from_lmax(lmax: int, parity: bool) -> Irreps:
#     """
#     Create irreps from lmax and parity.
#
#     Parameters:
#     -----------
#     lmax (int): maximum l value
#     parity (bool): whether to use parity
#     """
#     Irreps(
#             [
#                 (conv_feature_size, (l, p))
#                 for p in ((1, -1) if parity else (1,))
#                 for l in range(lmax + 1)
#             ]
#         )
