import torch

x = torch.tensor([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0],dtype=torch.float32)
y = torch.tensor([1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5],dtype=torch.float32)

w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)


def forward(x):
    return x*w

def loss(y_hat,y):
    return ((y_hat-y)**2).mean()

def gradient_backprop(x,y,y_hat):
    #dl/dw
    return torch.mul(2*x,y_hat-y).mean()

iters = 20
lr = 0.01
for epoch in range(iters):
        
    y_predicted = forward(x)
    l = loss(y_predicted,y)
    l.backward() #dl/dw
    
    with torch.no_grad():
      w -= lr * w.grad
    
    w.grad.zero_()
    print(f"epoch {epoch} ---> loss : {l}")
print(forward(20))
