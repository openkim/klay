import torch
from e3nn import o3


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


def get_torch_dtype(dtype):
    """
    Get the torch dtype from a string.
    Args:
        dtype (str): dtype string
    Returns:
        torch.dtype: torch dtype
    """
    if dtype == "float32":
        return torch.float32
    elif dtype == "float64":
        return torch.float64
    elif dtype == "float16":
        return torch.float16
    elif dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "int32":
        return torch.int32
    elif dtype == "int64":
        return torch.int64
    elif dtype == "int16":
        return torch.int16
    elif dtype == "int8":
        return torch.int8
    elif dtype == "long":
        return torch.long
    else:
        raise ValueError(f"Unknown dtype: {dtype}")
