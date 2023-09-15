import torch
from torchvision.datasets import GTSRB
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

BATCH_SIZE = 4

train_dataset = GTSRB(root='./gtsrb_dataset/',split="train",transform=ToTensor())


train_loader = DataLoader(dataset=train_dataset,shuffle=True)

for i,(input,label) in enumerate(train_loader):
    print(input[0].shape)
    plt.title(label)
    plt.imshow(input[0].T)
    plt.show()
    
