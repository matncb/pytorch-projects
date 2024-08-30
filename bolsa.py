import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional

empresa = 'VALE3.SA'    #PETR4.SA VALE3.SA
hist_treino = 5

#dados para treinar
inicio = dt.datetime(2012,1,1)
final = dt.datetime(2022,10,10)

dados = web.DataReader(empresa, 'yahoo', inicio, final)
dados = dados['Close'].values #Valor no fim do dia

def split_sequence(sequence, n_steps):
	x, y = list(), list()
	for i in range(len(sequence)):
		end_ix = i + n_steps
		if end_ix > len(sequence)-1:
			break
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		x.append(seq_x)
		y.append(seq_y)
	return np.array(x), np.array(y)

x_treinar, y_treinar = split_sequence(dados, hist_treino)
x_treinar = np.reshape(x_treinar, (x_treinar.shape[0], x_treinar.shape[1], 1))
#X = X.reshape((X.shape[0], X.shape[1], n_features))


#treinar modelo
modelo = Sequential()

'''
modelo.add(LSTM(units=50, return_sequences=True, input_shape=(x_treinar.shape[1], 1)))
modelo.add(Dropout(0.2))
modelo.add(LSTM(units=50, return_sequences=True))
modelo.add(Dropout(0.2))
modelo.add(LSTM(units=50))
modelo.add(Dropout(0.2))
modelo.add(Dense(units = 1)) #Prevendo o proximo valor
'''
modelo.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(x_treinar.shape[1], 1)))
modelo.add(Dense(1))

modelo.compile(optimizer = 'adam', loss = 'mean_squared_error')
modelo.fit(x_treinar, y_treinar, epochs = 100, batch_size = 32)


#teste previsão próximo dia

def grafico(precos_reais, previsao_precos):
    plt.plot(precos_reais, color ='red', label = f"Valor Real das acoes de {empresa}", marker = '.')
    plt.plot(previsao_precos, color="green", label = f"Previsao das acoes de {empresa}", marker = '.' )
    plt.title(f"{empresa} Preco Acao")
    plt.xlabel('Tempo')
    plt.ylabel(f"{empresa} Preco Acao")
    plt.legend()
    plt.show()

teste_inicio = dt.datetime(2022,10,10)
teste_final = dt.datetime.now()

dados_teste = web.DataReader(empresa, 'yahoo', teste_inicio, teste_final)
dados_teste = dados_teste['Close'].values

total_dados = []
for i in range(len(dados) - hist_treino, len(dados)):
    total_dados.append(dados[i])
for i in dados_teste:
    total_dados.append(i)

x_teste, y_teste = split_sequence(total_dados, hist_treino)
x_teste = np.reshape(x_teste, (x_teste.shape[0], x_teste.shape[1], 1))

previsao_precos = modelo.predict(x_teste)
grafico(y_teste, previsao_precos)