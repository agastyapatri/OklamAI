import numpy as np
import scipy as sp
import torch
import torch.nn as nn
torch.manual_seed(0)
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available(): torch.set_default_device("cuda")


class KDATA(torch.utils.data.Dataset):
    def __init__(self, PATH:str) -> None:
        super().__init__() 
        self.path = PATH 
        
    
    def __getitem__(self, i):
        return "KDATA"[:i]

    def __len__(self, ) -> int:
        pass 
        

if __name__ == "__main__":
    test = KDATA(PATH = )
    print(test[4])

