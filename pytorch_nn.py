import torch
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
import torch.nn as nn

x = torch.tensor([[1.0],[2.0],[3.0],[4.0],[5.0],[6.0],[7.0],[8.0],[9.0],[10.0],[11.0],[12.0],[13.0]],dtype=torch.float32)
y = torch.tensor([[1.5],[2.5],[3.5],[4.5],[5.5],[6.5],[7.5],[8.5],[9.5],[10.5],[11.5],[12.5],[13.5]],dtype=torch.float32)

x_numpy,y_numpy = datasets.make_regression(n_samples=100,n_features=1,noise=10,random_state=1)

X = torch.from_numpy(x_numpy.astype(np.float32))
Y = torch.from_numpy(y_numpy.astype(np.float32))
Y = Y.view(Y.shape[0],1)

n_samples,n_features = X.shape


class LinearReg(nn.Module):
    def __init__(self,in_features,out_features):
        super(LinearReg,self).__init__()
        self.linear = nn.Linear(in_features,out_features)
    def forward(self,x):
        return self.linear(x)
    


model = LinearReg(n_features,n_features)


iterations = 1000
learning_rate = 0.01
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)


for epoch in range(iterations):
    #forward propagation
    Y_predicted = model.forward(X)
   
    #loss
    l = loss(Y_predicted,Y)
    #backward propagation
    l.backward()
    #weights update
    optimizer.step()

    #clear the gradients
    optimizer.zero_grad()

    print(f'epoch {epoch} ---> loss : {l}')



prediction = model.forward(X).detach()
print(prediction)
plt.plot(X,Y,'o')
plt.plot(X,prediction,color='green')
plt.show()

