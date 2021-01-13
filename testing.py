import torch
from torch.optim import AdamW, SGD
from models import Net, NeuralMemory
from time import sleep

net = Net(4, 2, 3) # has 4 layers, the model input is of size 2, output is of size 3
# net = NeuralMemory(2, 3)
optimizer = SGD(lr=0.03, params=net.parameters())
criterion = torch.nn.MSELoss()

x = torch.tensor([0.4, 0.9], dtype=float)
y = torch.tensor([0.3, 0.5, 0.9], dtype=float)

while True:
    optimizer.zero_grad()
    out = net(x.detach())
    print(out.shape)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    print(loss.detach())
    sleep(0.03)