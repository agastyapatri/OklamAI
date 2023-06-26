# **OklamAI**
_Training a Neural Network on Kendrick Lamar's lyrics._

steps to prepare a text corpus for ingestion:
   
~~1. Tokenization : the process of breaking the text into individual words or "tokens". This can be done using regular expressions or natural language processing libraries~~
   

~~2. Standardization: This is the process of standardizing the tokens. This can involve removing stop words, stemming or lemmatizing words, and converting all words into lowercase. 
    
3. Encoding: This is the process of converting the tokens into numerical values. This can be done using a bag-of-words, word embedding etc. 
   
4. Padding: This is the process of adding padding to the encoded text. This is necessary to ensure that all the text samples have all the same length. 

## **Notes**

At this stage, I'm working on getting an implementation working; the finetuning will come later 

* I've opted for one-hot encoding at this stage


Steps to create the model:  
    1. Collect a dataset of lyrics
    2. Clean and preprocess the data 
    3. Choose a neu ral network architecture:    LSTM, GRU, Transformer
    4. Train the model
    5. Generate Lyrics

* The batch size dictates the length of the sequence. 


### **Text Generation with LSTM in PyTorch**
RNNs can be used for time series prediction (regression). They can also be used as generative models (classification).

A generative model learns certain pattern from data, such that it can generate a sequence in the same style as the learned data. Generative Adversarial Networks are a class of their own. Transformer models that use attention mechanisms are also useful to generate text passages. 







