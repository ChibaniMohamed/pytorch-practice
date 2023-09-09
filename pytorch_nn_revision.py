from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import SGD

batch_size = 4
learning_rate = 0.01
epochs = 100

dataset = MNIST(root='./data',train=True,transform=ToTensor())
dataloader = DataLoader(dataset=dataset,shuffle=True,batch_size=batch_size)

class NeuralNet(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_size,128)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(128,48)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(48,output_size)
    def forward(self,input):
        flatten = self.flatten(input)
        linear1 = self.linear1(flatten)
        activation1 = self.activation1(linear1)
        linear2 = self.linear2(activation1)
        activation2 = self.activation2(linear2)
        return self.linear3(activation2)

model = NeuralNet(28*28,10)
optimizer = SGD(params=model.parameters(),lr=learning_rate)
loss = nn.CrossEntropyLoss()

for i in range(500):
    l = 0
    for batch in range(batch_size-1):
        input,label = next(iter(dataloader))
       
        
        prediction = model.forward(input[batch][0])
        l = loss(prediction,label[batch])
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    print('epoch ---> ',i+1,' || ','loss ---> ',l)





for i in range(epochs):
    input,label = next(iter(dataloader))
    for batch in range(batch_size):
        
        prediction = model.forward(input[batch])
        prediction = prediction.argmax(1)
        plt.title(prediction)
        plt.imshow(input[batch].view(28,28),cmap='gray')
        plt.show()
