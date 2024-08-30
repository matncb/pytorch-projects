import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
import yfinance as yf

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

empresas= ['RADL3.SA', 'KLBN11.SA', 'BBSE3.SA', 'ENBR3.SA', 'ITSA4.SA', 'USIM5.SA', 'CMIG4.SA', 'BBDC3.SA', 'BRKM5.SA', 'JBSS3.SA', 'BBDC4.SA', 'BRML3.SA', 'OIBR4.SA', 'BBAS3.SA',
            'ECOR3.SA', 'CCRO3.SA', 'VALE5.SA', 'UGPA3.SA', 'EQTL3.SA',  'SBSP3.SA', 'LREN3.SA',  'MULT3.SA', 'PETR4.SA', 'PETR3.SA']    

def split_sequence(sequence, n_steps):
	x, y = list(), list()
	for i in range(len(sequence)):
		end_ix = i + n_steps
		if end_ix > len(sequence)-1:
			break
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		x.append(seq_x)
		y.append(seq_y)
	return x, y

#hyper parameter
sequence_lenght = 10
input_size = 1
hidden_size = 100
num_layers = 2

num_epochs = 50
lr = 0.001

#dados para treinar
inicio = dt.datetime(2012,1,1)
final = dt.datetime(2020,10,10)

x = []
y = []
for i in empresas:
    dados = yf.download(i, inicio, final)
    dados = dados['Close'].values

    xi, yi = split_sequence(dados, input_size)
    x += xi
    y += yi
x = np.array(x)
y = np.array(y)

x = torch.from_numpy(x).float()
y = torch.tensor([[i] for i in y]).float()


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)  # rnn, gru, lstm
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.R = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, 1)
    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        #out, _ = self.rnn(x, h0)
        out, _ = self.rnn(x, (h0,c0))
        out = out[:, -1, :]
        out = self.R(self.l1(out))
        out = self.l2(out)
        return out


def train_model(f, num_epochs):
    opt = torch.optim.Adam(f.parameters(), lr=lr)  #Adam, SGD
    L = nn.MSELoss()

    x = x.reshape(-1, sequence_lenght, input_size).to(device)

    # Train model
    for epoch in range(num_epochs):
        losses = []
        
        opt.zero_grad()
        loss_value = L(f(x), y) 
        loss_value.backward()
        opt.step()

        losses.append(loss_value.item())
        print('Epoch ' + str(epoch + 1) + ':' + ' loss: ' + str(losses[-1]))
    return f

f = NeuralNet(input_size, hidden_size).to(device)
f = train_model(f, num_epochs)

#dados para testar
inicio = dt.datetime(2021,1,1)
final = dt.datetime(2022,10,10)

dados_teste = yf.download(empresas[0], inicio, final)
dados_teste = dados_teste['Close'].values

x_teste, y_teste = split_sequence(dados_teste, input_size)
x_teste = torch.from_numpy(np.array(x_teste)).float()
y_teste = torch.from_numpy(np.array(y_teste)).float()

with torch.no_grad():
    plt.plot(y_teste, color = 'green')
    plt.plot(f(x_teste), color = 'red')
    plt.show()