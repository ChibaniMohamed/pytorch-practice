import torch


x = torch.tensor([2.0],requires_grad=True)
with torch.no_grad():
    y = x*10
    print(y)
print(x)