import torch
import torch.nn as nn
from torchvision.datasets import GTSRB
from torch.utils.data import DataLoader
from torch.optim import SGD
from torchvision.transforms import ToTensor,Resize,Compose
import matplotlib.pyplot as plt
import tqdm
BATCH_SIZE = 4
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
transforms = Compose([
Resize([112,112]),
ToTensor()
])

train_dataset = GTSRB(root='./gtsrb_dataset/',split="train",transform=transforms)
train_loader = DataLoader(dataset=train_dataset,shuffle=True)


class GTSRB_NETWORK(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(GTSRB_NETWORK,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_dim,245)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(245,128)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(128,80)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(80,output_dim)
        
    def forward(self,input):
        flatten = self.flatten(input)
        hidden_layer1 = self.linear1(flatten)
        activation_fn1 = self.relu1(hidden_layer1)
        hidden_layer2 = self.linear2(activation_fn1)
        activation_fn2 = self.relu2(hidden_layer2)
        hidden_layer3 = self.linear3(activation_fn2)
        activation_fn3 = self.relu3(hidden_layer3)
        output = self.linear4(activation_fn3)
        return output
    
EPOCHS = 2
LEARNING_RATE = 0.01
INPUT_DIM = 3*112*112
OUTPUT_DIM = 43
STEPS = len(train_loader)
model = GTSRB_NETWORK(INPUT_DIM,OUTPUT_DIM).to(device)
optimizor = SGD(params=model.parameters(),lr=LEARNING_RATE)
loss = nn.CrossEntropyLoss()
try:
    for epoch in range(EPOCHS):
        with tqdm.trange(STEPS) as STEPS_:
            for step,(input,label) in enumerate(train_loader):
                input = input.to(device)
                label = label.to(device)
                prediction = model.forward(input)
                l = loss(prediction,label)
                l.backward()
                optimizor.step()
                optimizor.zero_grad()
                STEPS_.colour = 'green'
                STEPS_.desc = f'Epoch [{epoch}/{EPOCHS}], Step [{step}/{STEPS}], Loss [{l}]'
                STEPS_.update(1)
    torch.jit.script(model).save('./gtsrb_model.pt')
except KeyboardInterrupt:
    torch.jit.script(model).save('./gtsrb_model.pt')


for i,(input,label) in enumerate(train_loader):
    
    normalized_input = input[0].permute(1,2,0)
    prediction = model.forward(input.to(device)).argmax(1)[0]
    plt.title(f'predicted :{prediction} | ground truth : {label[0]}')
    plt.imshow(normalized_input)
    plt.show()
    
