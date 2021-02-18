import torch
import torch.nn as nn
from collections import Counter
import numpy as np
import faiss

class NeuralMem2(nn.Module):
    """"
        in_features -> number of input features and output features because we will just retrieve the most similar pattern/tensor
        num_patterns -> number/count of stored patterns

        compares input query with the patterns stored in memory parameter and returns the pattern/parameter with the highest similarity value
        the variable 'out_features' is the number of stored patters.
        the variable 'in_features' is the size or shape of one pattern.

    """
    def __init__(self, in_features: int, num_patterns: int):
        super(NeuralMem2, self).__init__()
        self.memory = nn.Parameter(torch.rand(num_patterns, in_features), requires_grad=True)

    def forward(self, query):
        query = query.unsqueeze(0)
        similarities = torch.cosine_similarity(query, self.memory, 1)
        amax = torch.argmax(similarities)
        out = self.memory[amax]
        return out

class NeuralDictionary(nn.Module):
    """"
        in_features -> number of input features, size of input tensor,query
        out_features -> number of output features, size of output tensor
        num_patterns -> number/count of stored patterns/keys

        compares input query with the patterns/keys stored in "self.keys" and return the value from "self.values" associated to the key that is most similar to the input query.
          First compare the input query with all keys, we get the similarity values, get the index value/position of the highest/greatest similarity value, return the value from "self.values" at that index position.

        the variable 'in_features' is the size or shape of one pattern.

    """
    def __init__(self, in_features: int, out_features: int, num_patterns: int):
        super(NeuralDictionary, self).__init__()
        self.keys = nn.Parameter(torch.rand(num_patterns, in_features), requires_grad=True)
        self.values = nn.Parameter(torch.rand(num_patterns, out_features))

    def forward(self, query):
        query = query.unsqueeze(0)
        similarities = torch.cosine_similarity(query, self.keys, 1)
        amax = torch.argmax(similarities)
        out = self.values[amax]
        return out
