import torch
from torch.optim import AdamW, SGD
from models import Net, NeuralMemory
from time import sleep

net = Net(2, 2, 3) # has 4 layers, the model input is of size 2, output is of size 3
# net = NeuralMemory(2, 3)
optimizer = SGD(lr=0.03, params=net.parameters())
criterion = torch.nn.MSELoss()

x_train = [torch.rand(2) for i in range(10)]
y_train = [torch.rand(3) for i in range(10)]

while True:
    optimizer.zero_grad()
    loss = torch.tensor([0],dtype=float)
    for x,y in zip(x_train, y_train):
        out = net(x.detach())
        loss += criterion(out, y)

    loss.backward()
    optimizer.step()
    print(loss.detach())
    sleep(0.02)

