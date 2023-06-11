import torch

import torch.nn as nn

x = torch.tensor([[1.0],[2.0],[3.0],[4.0],[5.0],[6.0],[7.0],[8.0],[9.0],[10.0],[11.0],[12.0],[13.0]],dtype=torch.float32)
y = torch.tensor([[1.5],[2.5],[3.5],[4.5],[5.5],[6.5],[7.5],[8.5],[9.5],[10.5],[11.5],[12.5],[13.5]],dtype=torch.float32)

in_features,out_features = x.shape


class LinearReg(nn.Module):
    def __init__(self,in_features,out_features):
        super(LinearReg,self).__init__()
        self.linear = nn.Linear(in_features,out_features)
    def forward(self,x):
        return self.linear(x)
    





model = LinearReg(out_features,out_features)

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

iters = 2500
for epoch in range(iters):
    
    y_pred = model.forward(x)
    l = loss(y,y_pred)
    
    l.backward()
    
    optimizer.step()

    optimizer.zero_grad()
    print(f"epoch {epoch} ---> loss : {l}")

print(model(torch.tensor([29.0])))

