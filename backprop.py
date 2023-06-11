import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0,requires_grad=True)
b = torch.tensor(3.0)
#forward pass
y_hat = x*w+b
#compute the loss
loss = (y_hat-y)**2

loss.backward() #backward pass (dloss/dw chain rule) 

print(w.grad) #result of backpropagation


