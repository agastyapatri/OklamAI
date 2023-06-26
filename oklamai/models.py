import numpy as np
import scipy as sp
import torch
import torch.nn as nn
torch.manual_seed(0)
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available(): torch.set_default_device("cuda")

class oklamai_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.net = nn.LSTM(input_size, hidden_size)

    def forward(self, x):
        h0 = torch.zeros(1, 1, self.hidden_size)
        c0 = torch.zeros(1, 1, self.hidden_size)
        outputs, (h, c) = self.net(x, (h0, c0))
        return outputs 

if __name__ == "__main__":
    network = oklamai_LSTM(10, 20)
    x = torch.randn(1, 1, 10)
    outputs = network(x)