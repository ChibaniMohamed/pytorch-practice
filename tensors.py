import torch
import platform
import math
if torch.cuda.is_available():
    device = torch.device('cuda')
    
    x = torch.tensor([1.,2.],requires_grad=True)
    y = torch.tensor([3.,4.],requires_grad=True)
    z = x**2 + y**2
    print(z)
    external_grad = torch.tensor([1., 1.])
    print(x.grad)
    #z.backward(gradient=external_grad) #dy/dx
print(x)
print(x.requires_grad_(False))
print(x)