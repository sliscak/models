import torch
import torch.nn as nn
from collections import Counter

class NeuralDictionary(nn.Module):
     # Compares a query againts all keys a produces a confidence/probability for each key, the confidence/probability is multiplied by the value and summed up.
    
     # The model could be speed up with similarity search, or by learning just the Top highest probability/confidence keys or values(or both).
     # We could track which key-value pairs have been learned(with a list of counters or the Counter object) and use that to tell how surprised the network is to see a particular query(or state in Reinforcement Learning).
     #   That could be very useful in Reinforcement Learning(as curiosity value) or Classification to detect which class has not been learned.
     #   In Reinforcemenet learning the count of 0 would represent the highest/maximum curiosity/uncertainty value. That would represent a state(or location) that has not been visited. An agent guided by curiosity would try to visit and learn that state.
     #   If the key would represent a Class the count of 0 would suggest that that particular Class has not been learned. 
     #   So byt tracking the count of used(or top confidence) key-value pairs while learning we would learn the uncertainty(or curiosity) values.
     #   Key-value pairs that have not been learned while Trainig the model, that is their attention/confidence value was 0.
     #   The query could be resized (like an image) to lower the computational requirements.

    def __init__(self):
        super(NeuralDictionary, self).__init__()
        # 500 keys each of size 100, so the query needs to be of size 100
        self.keys = nn.Parameter(torch.randn(500, 100, dtype=torch.double))
        
        # 500 values each of size 4, the output of the model will be of size 4
        self.values = nn.Parameter(torch.randn(500, 4, dtype=torch.double))

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        attention = torch.matmul(self.keys, query)
        attention = torch.softmax(attention, 0)
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        #out = torch.sigmoid(out)
        
        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        #print(self.meta.most_common(10))
      
        return out
    
class NeuralDictionaryV2(nn.Module):

    def __init__(self):
        super(NeuralDictionaryV2, self).__init__()
        # Keys are replaced with Linear layer because they are equivalent
        # 500 keys each of size 100, so the query needs to be of size 100
        self.linear = nn.Linear(100, 500)

        # 500 values each of size 4, the output of the model will be of size 4
        self.values = nn.Parameter(torch.randn(500, 4, dtype=torch.double))

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        attention = self.linear(query)
        attention = torch.softmax(attention, 0)
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out

class NeuralDictionaryV3(nn.Module):
    # Going Full Circle.
    def __init__(self):
        super(NeuralDictionaryV3, self).__init__()
        # Keys are replaced with Linear layer because they are equivalent
        # 500 keys each of size 100, so the query needs to be of size 100
        self.linear = nn.Linear(100, 500)

        # Values are replaced with Linear layer because they are equivalent
        # 500 values each of size 4, the output of the model will be of size 4
        self.linear2 = nn.Linear(500, 4)

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        attention = self.linear(query)
        attention = torch.softmax(attention, 0)
        print(attention.shape)
        out = self.linear2(attention)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out
     
class NeuralDictionaryV4(nn.Module):

    def __init__(self):
        super(NeuralDictionaryV4, self).__init__()
        # 500 keys each of size 100, so the query needs to be of size 100
        self.keys = nn.Parameter(torch.randn(500, 100, dtype=torch.float))

        # 500 values each of size 4, the output of the model will be of size 4
        self.values = nn.Parameter(torch.randn(500, 4, dtype=torch.float))

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        # attention = torch.matmul(self.keys, query)
        query = torch.unsqueeze(query, 0)
        query = query.repeat(500, 1) # now query has shape (500,100)
        attention = torch.abs(self.keys - query) # computes absolute difference per element
        attention = -torch.sum(attention, 1) # computes absolute difference per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0) # compute the probabilities from the differences
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out
