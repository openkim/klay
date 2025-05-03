import math

import torch


def ShiftedSoftPlus(x):
    return torch.nn.functional.softplus(x) - math.log(2.0)
