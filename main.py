import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import os 
torch.manual_seed(0)
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available(): torch.set_default_device("cuda")

from oklamai.dataset import KDATA
from oklamai.models import oklamai_LSTM
from oklamai.traintest import Trainer


#   Hyperparameters
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
SEQ_LEN = 8 


#   1. Loading the data 
dataset = KDATA(PATH = "data/KDOT.txt")
dataloader = torch.utils.data.DataLoader(dataset, batch_size = SEQ_LEN, shuffle=False)

#   2. Loading the network 
model = oklamai_LSTM(input_size=dataset.dim, hidden_size=2000)
testdata = torch.randn(1, 1, 7508)


#   3. Training the model 
trainer = Trainer(model, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)

#--------------------------------------------------#
#   Write the training loop and get dumb baselines.
#--------------------------------------------------#



