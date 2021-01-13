import torch
import torch.nn as nn
from collections import Counter
import numpy as np
import faiss

# NEURAL DICTIONARY

class NeuralDictionary(nn.Module):
     # Compares a query againts all keys and produces a confidence/probability for each key, the confidence/probability is then multiplied by the value and summed up.
     # Somehow this is looks similar to Q-learning where we search for the state(position) in a table and update the values associated to the state.    

     # The model could be speed up with similarity search, or by learning just the Top highest probability/confidence keys or values(or both).
     # We could track which key-value pairs have been learned(with a list of counters or the Counter object) and use that to tell how surprised the network is to see a particular query(or state in Reinforcement Learning).
     #   The per key counter could be increased by 1 for every time the key-value pair was trained or every time the key-value pair was trained and the key confidence/probability/attention was the highest from all keys ( or was in the Top-K list),
     #    or if the key-value pair has succesfully classified an object/state or other tensor.
     #   So the per key counter could be seen as a king of flag that tells us if the correspondig key-value pair has been learned or is completely random.
     #   That could be very useful in Reinforcement Learning(as curiosity value) or Classification to detect which class has not been learned.
     #   In Reinforcemenet learning the count of 0 would represent the highest/maximum curiosity/uncertainty value. That would represent a state(or location) that has not been visited. An agent guided by curiosity would try to visit and learn that state.
     #   If the key would represent a Class the count of 0 would suggest that that particular Class has not been learned. 
     #   So byt tracking the count of used(or top confidence) key-value pairs while learning we would learn the uncertainty(or curiosity) values.
     #   Key-value pairs that have not been learned while Trainig the model, that is their attention/confidence value was 0.
     #   The query could be resized (like an image) to lower the computational requirements.

    def __init__(self, in_features: int, out_features: int, capacity: int):
        super(NeuralDictionary, self).__init__()
        # capacity represents the number of key-value pairs (or just the number of keys if there are no values).
        self.capacity = capacity
        # {capacity} keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example: 500 keys each of size 100, so the query needs to be of size 100
        self.keys = nn.Parameter(torch.randn(capacity, in_features, dtype=torch.float))

        # C values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        self.values = nn.Parameter(torch.randn(capacity, out_features, dtype=torch.float))

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
    # Going Full Circle. HERE I REALISED THAT THE NEURAL DICTIONARY IS JUST A TWO LAYER NEURAL NETWORK(OR MLP) WITH A SOFMTAX ACTIVATION FUNCTION BETWEEN THEM.
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
     # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key, 
     #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.
      
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
        attention = torch.abs(self.keys - query) # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        attention = -torch.sum(attention, 1) # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0) # compute the probabilities from the differences
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out
     
class NeuralDictionaryV5(nn.Module):
    # Compares all keys with the query, computes the absolute differences per element between key and query, sums up the differences per key,
    #    then uses softmax to compute the probabilities per key and matrix multiplies the probabilities with the values.

    def __init__(self, in_features: int, out_features: int, capacity: int):
        super(NeuralDictionaryV5, self).__init__()
        # capacity,C, represents the number of key-value pairs.
        self.capacity = capacity
        # C keys each of size {in_features}, so the query needs to be of size {in_features}
        # for example 500 keys each of size 100, so the query needs to be of size 100
        self.keys = nn.Parameter(torch.randn(capacity, in_features, dtype=torch.float))

        # C values each of size {out_features}, the output of the model will be of size {out_features}
        # for example 500 values each of size 4, the output of the model will be of size 4
        self.values = nn.Parameter(torch.randn(capacity, out_features, dtype=torch.float))

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        # attention = torch.matmul(self.keys, query)
        query = torch.unsqueeze(query, 0)
        query = query.repeat(self.capacity, 1)  # now query has shape (capacity, in_features)  , for example (500, 100)
        attention = torch.abs(self.keys - query)  # computes absolute difference per element , (maybe later try euclidean distance or cosine similarity)
        attention = -torch.sum(attention,1)  # computes the sum of absolute differences per key, that is one key has one value, and makes it negative because of the following softmax operation
        attention = torch.softmax(attention, 0)  # compute the probabilities from the differences
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out
     
class NeuralDictionaryV6(nn.Module):

    def __init__(self):
        super(NeuralDictionaryV6, self).__init__()
        self.keys = None
        self.values = None

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        self.meta = Counter()

    def forward(self, query):
        attention = torch.matmul(self.keys, query)
        attention = torch.softmax(attention, 0)
        #attention = (attention >= torch.max(attention)) * attention
        out = torch.matmul(attention, self.values)
        # use a activation function here if you want, like sigmoid, but that depends on the task, the output range we need
        # out = torch.sigmoid(out)

        amax = torch.argmax(attention)
        self.meta.update(f'{amax}')
        # print(self.meta.most_common(10))

        return out

    def update(self, key, value):
        # add key as trainable parameter to the dictionary and value as nontrainable paramater/tensor to the dictionary 
        key = torch.unsqueeze(key, 0)
        value = torch.unsqueeze(value, 0)
        if self.values is None:
            self.keys = nn.Parameter(key)
            self.values = value * 1
        else:
            self.keys = nn.Parameter(torch.cat((self.keys, key)))
            # print(self.keys.shape)
            self.values = torch.cat((self.values, value), 0)
               

# IDEA: Save all input quries as nontrainable keys and associate values with them (every saved nontrainable key(the saved query) would have a trainable/learnable value associated)
#    The algorithm would use similarity search to find the most similar key and output the value * confidence(the confidence is from similarity search, the similarity or difference between query and key).
#    The algorithm search over all keys, returns their confidence/probabilities, uses that as attention and multiplies the values with their corresponding attention and sums up all values into the final tensor.
#    Example: The algorithm would save an image of a STOP sign as a key and associate a random learnable parameter as value to the key, thus creating a key-value pair. (key --> nontrainable, value --> trainable)
#              If the algorithm gets a query(in this example an image) as input, it compares/similarity-searches all keys with that query. The similarity or differences would be softmaxed that is their range would be
#                   would be limited to be between 0 and 1. ..... A 1(one) would mean that the key a query are indentical/the same. A 0(zero) would mean that the query is different from the key.
#              So the algorithm creates a tensor of Attention(values)  and then matrix multiplies them with the values associated to the keys, thus creating the final tensor.
#              While trainining the values associated to the keys would be learned. So the algorithm/agent could learn to stop at the STOP sing, change the direction or do something else.

class NeuralDictionaryV7(nn.Module):
    # Dictionary where the key is static(nontrainable) and the value is a learnable(trainable) parameter.
    # All keys are saved inside the index.
    # Use the update method to add key-value pairs.
    # The next evolution would be to turn the keys into learnable(trainable) parameters.
    def __init__(self, in_features: int):
        super(NeuralDictionaryV7, self).__init__()
        self.values = None
        self.index = faiss.IndexFlatL2(in_features)

        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        # self.meta = Counter()

    def forward(self, query):
        q = torch.unsqueeze(query, 0).detach().numpy().astype('float32')
        distances, ids = self.index.search(q, 1)
        print(f'IDS: {ids}')
        id = ids[0]
        out = torch.matmul(self.values[id], query)
        return out

    def update(self, key, value):
        # all keys and values should be of the same shape
        key = torch.unsqueeze(key, 0)
        self.index.add(key.detach().numpy().astype('float32'))
        value = torch.unsqueeze(value, 0)
        print(value)
        if self.values is None:
            self.values = nn.Parameter(value)
        else:
            self.values = nn.Parameter(torch.cat((self.values, value)))
               
class NeuralDictionaryV8(nn.Module):
    # Dictionary where the key is static(nontrainable) and the value is a learnable(trainable) parameter.
    # All keys are saved inside the index.
    # Use the update method to add key-value pairs.
    # Keys and values can be of a different size(shape) but all subsequent keys and values must be of the same size(shape) as the first key and value respectively.
    # Returns the value from the key-value pair, for which the key is most similar to the query.
    # (the algorithm finds the most similar key to a query and then returns the value of the key-value pair which had the most similar key.)

    def __init__(self, in_features: int):
        super(NeuralDictionaryV8, self).__init__()
        self.values = None
        self.index = faiss.IndexFlatL2(in_features)
        # to track and later see how many times a key has been chosen as the most important one(the key with the highest confidence)
        # self.meta = Counter()

    def forward(self, query):
        q = torch.unsqueeze(query, 0).detach().numpy().astype('float32')
        distances, ids = self.index.search(q, 1)
        print(f'IDS: {ids}')
        id = ids[0]
        return self.values[id]

    def update(self, key, value):
        key = torch.unsqueeze(key, 0)
        self.index.add(key.detach().numpy().astype('float32'))
        value = torch.unsqueeze(value, 0)
        print(value)
        if self.values is None:
            self.values = nn.Parameter(value)
        else:
            self.values = nn.Parameter(torch.cat((self.values, value)))


class NeuralMemory(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super(NeuralMemory, self).__init__()
        self.memory = nn.Parameter(torch.rand(out_features, in_features), requires_grad=True)

    def forward(self, query):
        query = query.unsqueeze(0)
        out = torch.cosine_similarity(query, self.memory, 1)
        return out


nm = NeuralMemory(3, 10)
a = torch.tensor([2,3,1])
print(a.shape)
o = nm(a)
print(o)
print(o.shape)

