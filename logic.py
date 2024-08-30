import numpy as np

x_data = np.array([1,2,3,4], dtype =np.float32)
y_data = np.array([2,4,6,8], dtype = np.float32)

w = 0.0

#prediction
def forward(x):
    return w*x

#loss
def loss(y, y_predicted):
    return ((y_predicted -y)**2).mean()

#gradient
#MSE = 1/N * (w*x - y)**2
#dJ/dw = 1/N * 2 * x * (w*x -y)

def gradient(x,y,y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print('Prediction before training: f(5) = ' + str(forward(5)))

#Training
lr = 0.01
iters = 10

for epochs in range(iters):
    y_pred = forward(x_data)
    l = loss(y_data, y_pred)
    w_grad = gradient(x_data, y_data, y_pred)

    w -= lr*w_grad
    print('Epoch ' + str(epochs + 1) +':' ' w = ' + str(w) + ' loss = ' + str(l))

print('Prediction after training: f(5) = ' + str(forward(5)))
