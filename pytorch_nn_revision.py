from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import SGD
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device used : ',device)
batch_size = 4
learning_rate = 0.01
epochs = 100

train_data = MNIST(root='./data',train=True,transform=ToTensor())
test_data = MNIST(root='./data',train=False,transform=ToTensor())

train_dataloader = DataLoader(dataset=train_data,shuffle=True,batch_size=batch_size)
test_dataloader = DataLoader(dataset=test_data,shuffle=True)

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

model = NeuralNet(28*28,10).to(device)
optimizer = SGD(params=model.parameters(),lr=learning_rate)
loss = nn.CrossEntropyLoss()

last_epoch = 0
last_step = 0
last_loss = 0
try : 
    for epoch in range(epochs):
        last_epoch = epoch
        l = 0
        
        for i,(input,label) in enumerate(train_dataloader):
            last_step = i
            input = input.to(device)
            label = label.to(device)
            prediction = model.forward(input)
            l = loss(prediction,label)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            last_loss = l
            if (i+1) % 100 == 0 :
                print(f'Epoch [{epoch+1}/{epochs}], Step {i+1}/{len(train_dataloader)}, Loss [{l}]')
except KeyboardInterrupt:
    print('saving checkpoint ...')
    torch.save({
        'epoch':last_epoch,
        'step':last_step,
        'loss':last_loss,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict()
    },'./model_checkpoint.tar')
    print('checkpoint saved !')
    print('saving model ...')  
    torch.jit.script(model).save('./model.pt')   
    print('model saved !')  
