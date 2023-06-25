import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import os 
torch.manual_seed(0)
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available(): torch.set_default_device("cuda")


class KDATA(torch.utils.data.Dataset):
    """
        The Kendrick Lamar datset. 
        
        [args]: 
            PATH:str - location of the plain text data 
    """
    def __init__(self, PATH:str) -> None:
        super().__init__() 
        self.path = PATH 
        self.lyrics = None 
        with open(PATH, "r") as f:
            self.lyrics = f.readlines()

    
    def __getitem__(self, i) -> torch.Tensor:
        return self.lyrics[i]

    def __len__(self, ) -> int:
        return len(self.lyrics) 
    
    def prepare_data(self, ):
        """
            Function to convert plain text into a format which is suitable as an input for the neural network model

        """
        pass 
        

if __name__ == "__main__":
    test = KDATA(PATH = "data/KDOT.txt")
    for i in range(len(test)):
        print(test[i])

        if i == 49: 
            break 


    