import torch
import torch.nn.functional

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode



@compile_mode("script")
class ElectronicConfigurationEncoding(torch.nn.Module):
    """
    Compute a binary encoding of atoms' discrete electronic configurations.
    Based on Spookynet's implementation. 
    Z 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p vs vp vd vf
    """
    def __init__(self):
        super().__init__()
        self.irreps_out = Irreps([(24, (0, 1))])
        e_config = torch.tensor([#Z    1s 2s 2p 3s 3p 3d  4s 4p 4d  5s 5p 4f  5d  6s 6p 5f  6d  7s 7p vs vp vd   vf 
                                [1,   1, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 1, 0, 0,   0],
                                [2,   2, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 0, 0,   0],
                                [3,   2, 1, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 1, 0, 0,   0],
                                [4,   2, 2, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 0, 0,   0],
                                [5,   2, 2, 1, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 1, 0,   0],
                                [6,   2, 2, 2, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 2, 0,   0],
                                [7,   2, 2, 3, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 3, 0,   0],
                                [8,   2, 2, 4, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 4, 0,   0],
                                [9,   2, 2, 5, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 5, 0,   0],
                                [10,  2, 2, 6, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 6, 0,   0],
                                [11,  2, 2, 6, 1, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 1, 0, 0,   0],
                                [12,  2, 2, 6, 2, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 0, 0,   0],
                                [13,  2, 2, 6, 2, 1, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 1, 0,   0],
                                [14,  2, 2, 6, 2, 2, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 2, 0,   0],
                                [15,  2, 2, 6, 2, 3, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 3, 0,   0],
                                [16,  2, 2, 6, 2, 4, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 4, 0,   0],
                                [17,  2, 2, 6, 2, 5, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 5, 0,   0],
                                [18,  2, 2, 6, 2, 6, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 6, 0,   0],
                                [19,  2, 2, 6, 2, 6, 1,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 1, 0, 0,   0],
                                [20,  2, 2, 6, 2, 6, 2,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 0, 0,   0],
                                [21,  2, 2, 6, 2, 6, 1,  2, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 0, 1,   0],
                                [22,  2, 2, 6, 2, 6, 2,  2, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 0, 2,   0],
                                [23,  2, 2, 6, 2, 6, 3,  2, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 0, 3,   0],
                                [24,  2, 2, 6, 2, 6, 5,  1, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 1, 0, 5,   0],
                                [25,  2, 2, 6, 2, 6, 5,  2, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 0, 5,   0],
                                [26,  2, 2, 6, 2, 6, 6,  2, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 0, 6,   0],
                                [27,  2, 2, 6, 2, 6, 7,  2, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 0, 7,   0],
                                [28,  2, 2, 6, 2, 6, 8,  2, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 0, 8,   0],
                                [29,  2, 2, 6, 2, 6, 10, 1, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 1, 0, 10,  0],
                                [30,  2, 2, 6, 2, 6, 10, 2, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 0, 10,  0],
                                [31,  2, 2, 6, 2, 6, 10, 2, 1, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 1, 10,  0],
                                [32,  2, 2, 6, 2, 6, 10, 2, 2, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 2, 10,  0],
                                [33,  2, 2, 6, 2, 6, 10, 2, 3, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 3, 10,  0],
                                [34,  2, 2, 6, 2, 6, 10, 2, 4, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 4, 10,  0],
                                [35,  2, 2, 6, 2, 6, 10, 2, 5, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 5, 10,  0],
                                [36,  2, 2, 6, 2, 6, 10, 2, 6, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 6, 10,  0],
                                [37,  2, 2, 6, 2, 6, 10, 2, 6, 1,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 1, 0, 0,   0],
                                [38,  2, 2, 6, 2, 6, 10, 2, 6, 2,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 0, 0,   0],
                                [39,  2, 2, 6, 2, 6, 10, 2, 6, 1,  2, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 0, 1,   0],
                                [40,  2, 2, 6, 2, 6, 10, 2, 6, 2,  2, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 0, 2,   0],
                                [41,  2, 2, 6, 2, 6, 10, 2, 6, 4,  1, 0, 0,  0,  0, 0, 0,  0,  0, 0, 1, 0, 4,   0],
                                [42,  2, 2, 6, 2, 6, 10, 2, 6, 5,  1, 0, 0,  0,  0, 0, 0,  0,  0, 0, 1, 0, 5,   0],
                                [43,  2, 2, 6, 2, 6, 10, 2, 6, 5,  2, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 0, 5,   0],
                                [44,  2, 2, 6, 2, 6, 10, 2, 6, 7,  1, 0, 0,  0,  0, 0, 0,  0,  0, 0, 1, 0, 7,   0],
                                [45,  2, 2, 6, 2, 6, 10, 2, 6, 8,  1, 0, 0,  0,  0, 0, 0,  0,  0, 0, 1, 0, 8,   0],
                                [46,  2, 2, 6, 2, 6, 10, 2, 6, 10, 0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 1, 0, 10,  0],
                                [47,  2, 2, 6, 2, 6, 10, 2, 6, 10, 1, 0, 0,  0,  0, 0, 0,  0,  0, 0, 1, 0, 10,  0],
                                [48,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 0, 0,  0,  0, 0, 0,  0,  0, 0, 2, 0, 10,  0],
                                [49,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 1, 0,  0,  0, 0, 0,  0,  0, 0, 2, 1, 10,  0],
                                [50,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 2, 0,  0,  0, 0, 0,  0,  0, 0, 2, 2, 10,  0],
                                [51,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 3, 0,  0,  0, 0, 0,  0,  0, 0, 2, 3, 10,  0],
                                [52,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 4, 0,  0,  0, 0, 0,  0,  0, 0, 2, 4, 10,  0],
                                [53,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 5, 0,  0,  0, 0, 0,  0,  0, 0, 2, 5, 10,  0],
                                [54,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 0,  0,  0, 0, 0,  0,  0, 0, 2, 6, 10,  0],
                                [55,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 1,  0,  0, 0, 0,  0,  0, 0, 1, 0, 0,   0],
                                [56,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 2,  0,  0, 0, 0,  0,  0, 0, 2, 0, 0,   0],
                                [57,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 1,  2,  0, 0, 0,  0,  0, 0, 2, 0, 1,   0],
                                [58,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 1,  1,  2, 0, 0,  0,  0, 0, 2, 0, 1,   1],
                                [59,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 3,  2,  0, 0, 0,  0,  0, 0, 2, 0, 0,   3],
                                [60,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 4,  2,  0, 0, 0,  0,  0, 0, 2, 0, 0,   4],
                                [61,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 5,  2,  0, 0, 0,  0,  0, 0, 2, 0, 0,   5],
                                [62,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 6,  2,  0, 0, 0,  0,  0, 0, 2, 0, 0,   6],
                                [63,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 7,  2,  0, 0, 0,  0,  0, 0, 2, 0, 0,   7],
                                [64,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 7,  1,  2, 0, 0,  0,  0, 0, 2, 0, 1,   7],
                                [65,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 9,  2,  0, 0, 0,  0,  0, 0, 2, 0, 0,   9],
                                [66,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 10, 2,  0, 0, 0,  0,  0, 0, 2, 0, 0,  10],
                                [67,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 11, 2,  0, 0, 0,  0,  0, 0, 2, 0, 0,  11],
                                [68,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 12, 2,  0, 0, 0,  0,  0, 0, 2, 0, 0,  12],
                                [69,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 13, 2,  0, 0, 0,  0,  0, 0, 2, 0, 0,  13],
                                [70,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 2,  0, 0, 0,  0,  0, 0, 2, 0, 0,  14],
                                [71,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 1,  2, 0, 0,  0,  0, 0, 2, 0, 1,  14],
                                [72,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 2,  2, 0, 0,  0,  0, 0, 2, 0, 2,  14],
                                [73,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 3,  2, 0, 0,  0,  0, 0, 2, 0, 3,  14],
                                [74,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 4,  2, 0, 0,  0,  0, 0, 2, 0, 4,  14],
                                [75,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 5,  2, 0, 0,  0,  0, 0, 2, 0, 5,  14],
                                [76,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 6,  2, 0, 0,  0,  0, 0, 2, 0, 6,  14],
                                [77,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 7,  2, 0, 0,  0,  0, 0, 2, 0, 7,  14],
                                [78,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 9,  1, 0, 0,  0,  0, 0, 1, 0, 9,  14],
                                [79,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 1, 0, 0,  0,  0, 0, 1, 0, 10, 14],
                                [80,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 0, 0,  0,  0, 0, 2, 0, 10, 14],
                                [81,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 1, 0,  0,  0, 0, 2, 1, 10, 14],
                                [82,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 2, 0,  0,  0, 0, 2, 2, 10, 14],
                                [83,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 3, 0,  0,  0, 0, 2, 3, 10, 14],
                                [84,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 4, 0,  0,  0, 0, 2, 4, 10, 14],
                                [85,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 5, 0,  0,  0, 0, 2, 5, 10, 14],
                                [86,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 0,  0,  0, 0, 2, 6, 10, 14],
                                [87,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 1,  0,  0, 0, 1, 0, 0,   0],
                                [88,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 2,  0,  0, 0, 2, 0, 0,   0],
                                [89,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 1,  2,  0, 0, 2, 0, 1,   0],
                                [90,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 2,  2,  0, 0, 2, 0, 2,   0],
                                [91,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 2,  1,  2, 0, 2, 0, 1,   2],
                                [92,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 3,  1,  2, 0, 2, 0, 1,   3],
                                [93,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 4,  1,  2, 0, 2, 0, 1,   4],
                                [94,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 6,  2,  0, 0, 2, 0, 0,   6],
                                [95,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 7,  2,  0, 0, 2, 0, 0,   7],
                                [96,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 7,  1,  2, 0, 2, 0, 1,   7],
                                [97,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 9,  2,  0, 0, 2, 0, 0,   9],
                                [98,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 10, 2,  0, 0, 2, 0, 0,  10],
                                [99,  2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 11, 2,  0, 0, 2, 0, 0,  11],
                                [100, 2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 12, 2,  0, 0, 2, 0, 0,  12],
                                [101, 2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 13, 2,  0, 0, 2, 0, 0,  13],
                                [102, 2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 14, 2,  0, 0, 2, 0, 0,  14],
                                [103, 2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 14, 2,  1, 0, 2, 1, 0,  14],
                                [104, 2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 14, 2,  2, 0, 2, 0, 2,  14],
                                [105, 2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 14, 3,  2, 0, 2, 0, 3,  14],
                                [106, 2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 14, 4,  2, 0, 2, 0, 4,  14],
                                [107, 2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 14, 5,  2, 0, 2, 0, 5,  14],
                                [108, 2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 14, 6,  2, 0, 2, 0, 6,  14],
                                [109, 2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 14, 7,  2, 0, 2, 0, 7,  14],
                                [110, 2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 14, 9,  1, 0, 1, 0, 9,  14],
                                [111, 2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 14, 10, 1, 0, 1, 0, 10, 14],
                                [112, 2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 14, 10, 2, 0, 2, 0, 10, 14],
                                [113, 2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 14, 10, 2, 1, 2, 1, 10, 14],
                                [114, 2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 14, 10, 2, 2, 2, 2, 10, 14],
                                [115, 2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 14, 10, 2, 3, 2, 3, 10, 14],
                                [116, 2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 14, 10, 2, 4, 2, 4, 10, 14],
                                [117, 2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 14, 10, 2, 5, 2, 5, 10, 14],
                                [118, 2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 10, 2, 6, 14, 10, 2, 6, 2, 6, 10, 14]], dtype=torch.float32)
        self.register_buffer("e_config",e_config)

    def forward(self, x):
        representation =  self.e_config[x - 1].to(x.dtype).to(x.device)
        return representation