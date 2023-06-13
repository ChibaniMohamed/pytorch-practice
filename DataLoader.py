import torch
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt


dataset = torchvision.datasets.CIFAR10(root='./data',train=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset,batch_size=4,shuffle=True)
data = iter(dataloader)
batch_size = 4
n_iterations = 2
for epoch in range(1,n_iterations+1):
   for i,(inputs,labels) in enumerate(dataloader):
      for batch_idx in range(1,batch_size+1):
        print(f'epoch {epoch} --> step {batch_idx}/{batch_size}')
      if(i == 1):
            break
'''
for i in range(10):
    index =  torch.randint(low=0,high=10000,size=(1,))
    print(index)
    x,y = dataset[index]

    plt.imshow(x.permute(1,2,0))
    plt.show()
'''
