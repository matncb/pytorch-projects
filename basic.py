import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.tensor([[6,2], [5,2], [1,3], [7,6]]).float()
y_data = torch.tensor([[1],[5],[2],[5]]).float()

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(2,8,bias=False)
        self.Matrix2 = nn.Linear(8,1,bias=False)
    def forward(self,x):
        x = self.Matrix1(x)
        x = self.Matrix2(x)
        return x

f = NeuralNet()
L = nn.MSELoss()
opt = SGD(f.parameters(), lr = 0.001)

losses = []
epochs = 50

for _ in range(epochs):
    opt.zero_grad() # flush previous epoch's gradient
    loss_value = L(f(x_data), y_data) #compute loss
    loss_value.backward() # compute gradient
    opt.step() # Perform iteration using gradient above
    losses.append(loss_value.item())

print(f(x_data))
plt.plot(losses)
plt.show()