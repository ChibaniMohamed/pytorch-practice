import torch
import platform
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(device)
    print(platform.python_version())
    x = torch.rand(3,3,3)
    y = torch.rand(3,3,3)
    z = x.mul(y)
    print(z)