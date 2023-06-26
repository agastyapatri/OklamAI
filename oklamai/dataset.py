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

        self.encoding_map = self._one_hot_encoded()[1]
        self.tensor = torch.tensor(self._one_hot_encoded()[0], dtype=torch.float32)
        self.dim = self.tensor.shape[1]

    
    def __getitem__(self, i) -> torch.Tensor:
        return self.tensor[i]

    def __len__(self, ) -> int:
        return len(self.tensor) 
    
    def _tokenize_data(self, ) -> list:
        """
            Function to convert plain text into a format which is suitable as an input for the neural network model
        """
        excludes = ["[",  "]", "\n", "''", ",", ".", "!", ":", "?", ")", "(", "*", " "]

        #   cleaning the letters and re-adding to the corpus 
        cleaned_lyrics = []
        tokenized = []
        for i in range(len(self.lyrics)):
            sentence = self.lyrics[i]
            split_sentence = sentence.split(" ")
            for i in range(len(split_sentence)):
                word = split_sentence[i]
                new_word = "".join([char for char in word if char not in excludes])
                tokenized.append(new_word.lower())
                split_sentence[i] = new_word
            cleaned_lyrics.append(split_sentence)
        return tokenized
    
    def _one_hot_encoded(self, word=None, vocab_size=None):
        """
            Function to one-hot-encode the words in the tokeneized dataset
            
            [args]:
                word:str - the word to be encoded
                vocab_size - the size of the set of unique words in the vocabulary 
        """
        tokenized = self._tokenize_data()
        vocab = list(set(tokenized)) 
        encoding_map = {} 

        for i in range(len(vocab)):
            word = vocab[i]
            encoded = np.zeros(len(vocab))
            encoded[i] = 1
            encoding_map[word] = encoded

        encoded = [encoding_map[word] for word in tokenized]
        return np.array(encoded), encoding_map



    

                 
    

if __name__ == "__main__":
    test = KDATA(PATH = "data/KDOT.txt")


