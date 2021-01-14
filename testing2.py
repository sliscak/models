import torch
from torch.optim import AdamW, SGD
from models import Net, NeuralMemory, Net2, Net3, Net4, Net5
from time import sleep
import streamlit as st


# net = Net(4, 2, 3, z=5) # has 4 layers, the model input is of size 2, output is of size 3, layers in between have output size z=5

net = Net5(1, 2, 3, z=10)
# net = NeuralMemory(2, 3)
optimizer = SGD(lr=0.000003, params=net.parameters())
criterion = torch.nn.MSELoss()

# x_train = [torch.rand(2) for i in range(10)]
# y_train = [torch.rand(3) for i in range(10)]

x_train = [torch.rand(2) for i in range(10)]
y_train = [torch.rand(3) for i in range(10)]

col1, col2 = st.beta_columns(2)
weights1_ph = col1.empty()
weights2_ph = col2.empty()
# weights1_ph = st.empty()
# weights2_ph = st.empty()
loss_ph = st.empty()

old_weights1 = None
old_weights2 = None
for i in range(10000):
    optimizer.zero_grad()
    loss = torch.tensor([0],dtype=float)
    for x,y in zip(x_train, y_train):
        out = net(x.detach())
        loss += criterion(out, y)
        # print(f'INPUT:\t{x}\nOUTPUT:\t{out}\nGTRUTH:\t{y}')
        # print('---')

    # print(list(net.parameters()))
    loss.backward()
    optimizer.step()
    # print(loss.detach())
    loss_ph.write(f'loss: {loss.detach().numpy()}')

    weights1 = net.layers[0].memory.detach().numpy()
    weights2 = net.param.detach().numpy()

    if old_weights1 is None or old_weights2 is None:
        old_weights1 = weights1
        old_weights2 = weights2



    weights1_ph.table(weights1)
    weights2_ph.table(weights2)
    # weights_ph.dataframe(weights)

    # weights_ph.write([torch.tensor(param) for param in net.param])
    # print([param for param in net.param])

    # st.write(list(net.parameters()))
    sleep(1)

