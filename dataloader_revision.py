import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
dataset = MNIST(root='./data',train=True,transform=transforms.ToTensor())

batch_size = 4
dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
data ,labels = next(iter(dataloader))
for i in range(batch_size):
    plt.title(labels[i])
    plt.imshow(data[i].view(28,28),cmap='gray')
    plt.show()
