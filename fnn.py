import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameter
input_size = 784
num_classes = 10
hidden_size = 100

num_epochs = 10
batch_size = 100
lr = 0.001

#MNIST
train_dataset =torchvision.datasets.MNIST(root = './data', train = True, transform = transforms.ToTensor(),  download = True)
test_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,hidden_size)
        self.l3 = nn.Linear(hidden_size,num_classes)
        self.R = nn.ReLU()
    def forward(self,x):
        x = self.R(self.l1(x))
        x = self.R(self.l2(x))
        x = self.l3(x)
        return x

def train_model(f,train_loader, num_epochs):
    opt = torch.optim.Adam(f.parameters(), lr=lr)  #Adam, SGD
    L = nn.CrossEntropyLoss()

    # Train model
    for epoch in range(num_epochs):
        losses = []
        
        for i, (images, labels) in enumerate(train_loader):  
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)

            opt.zero_grad()
            loss_value = L(f(images), labels) 
            loss_value.backward()
            opt.step()

            losses.append(loss_value.item())
        loss = sum(losses)/len(losses)
        print('Epoch ' + str(epoch + 1) + ':' + ' loss: ' + str(loss))
    return f

def test_model(f, test_loader):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0

        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)

            outputs = f(images)
            _, predictions = torch.max(outputs, 1)

            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()

        acc = n_correct/n_samples
        return acc

def show_test_model():
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images_reshape = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = f(images_reshape)
        _, predictions = torch.max(outputs, 1)

        for i in range(len(images_reshape)):
            plt.imshow(images[i][0], cmap ='gray')
            plt.title(str(predictions[i].numpy()))
            plt.show()

model = NeuralNet(input_size, hidden_size, num_classes).to(device)
f = train_model(model, train_loader, num_epochs)
acc = test_model(f, test_loader)
print(acc)
show_test_model()

