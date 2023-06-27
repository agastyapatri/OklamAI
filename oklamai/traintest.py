import numpy as np
import torch
import torch.nn as nn
torch.manual_seed(0)
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available(): torch.set_default_device("cuda")

class Trainer:
    """
        Training the model
        [args]:
    """
    def __init__(
            self, 
            model:nn.Module,
            num_epochs:int,
            learning_rate:float, 
            ) -> None:
        self.optim = torch.optim.Adam(model.paramters())
        self.criterion = nn.CrossEntropyLoss()



        
    def _train_one_func(self, ) -> type:
        pass


    def train(self, ):
        pass 


if __name__ == "__main__":
    pass