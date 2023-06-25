import numpy as np
import scipy as sp
import torch
import torch.nn as nn
torch.manual_seed(0)
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available(): torch.set_default_device("cuda")

