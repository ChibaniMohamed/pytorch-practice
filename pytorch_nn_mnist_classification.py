import torch
from torch.optim import SGD
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mnist_train_dataset = datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor())
mnist_test_dataset = datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor())

dataloader_train = DataLoader(dataset=mnist_train_dataset,shuffle=True)
dataloader_test = DataLoader(dataset=mnist_test_dataset,shuffle=True)

class NeuralNet(nn.Module):
    def __init__(self,input_size,output_size):
        super(NeuralNet,self).__init__()
        self.flatten = nn.Flatten()
        self.input = nn.Linear(input_size,512)
        self.relu = nn.ReLU()
        self.hidden_layer1 = nn.Linear(512,128)
        self.hidden_layer2 = nn.Linear(128,output_size)
    def forward(self,data):
        out = self.flatten(data)
        out = self.input(out)
        out = self.relu(out)
        out = self.hidden_layer1(out)
        out = self.hidden_layer2(out)
        return out

model = NeuralNet(1*28*28,10).to(device=device)

learning_rate = 0.01
epochs = 10
optimizer = SGD(params=model.parameters(),lr=learning_rate)
loss = nn.CrossEntropyLoss()



for epoch in range(epochs):
    l = 0
    for step in range(500):
        data,output = next(iter(dataloader_train))
        data = data.to(device)
        output = output.to(device)
        predicted_output = model.forward(data)
        
        l = loss(predicted_output,output)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        

    print(f'epoch {epoch} --> loss {l}')
for i in range(50):
    data_,output = next(iter(dataloader_test))
    data = data_.to(device)
    output = output.to(device)
    predicted = model.forward(data)
    predicted = predicted.argmax(1).cpu().numpy()
    print(f'real : {output}, predicted : {predicted}')
    plt.imshow(data_.view(28,28),cmap='gray')
    plt.show()

