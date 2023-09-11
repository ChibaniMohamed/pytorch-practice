import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

device = ('cuda' if torch.cuda.is_available() else 'cpu')

test_data = MNIST('./data',train=False,transform=ToTensor())
test_dataloader = DataLoader(dataset=test_data,shuffle=True)

model = torch.jit.load('./model.pt').to(device)


for i in range(len(test_dataloader)):
        input,label = next(iter(test_dataloader))
        input = input.to(device)
        label = label.to(device)
        prediction = model.forward(input)
        prediction = prediction.argmax(1)
        plt.title(f'truth : {label[0]} | prediction : {prediction[0]}')
        plt.imshow(input.cpu().view(28,28),cmap='gray')
        plt.show()