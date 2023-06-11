import numpy as np

x = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0],dtype=np.float32)
y = np.array([1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5],dtype=np.float32)
w = 0.0


def forward(x):
    return x*w

def loss(y_hat,y):
    return ((y_hat-y)**2).mean()

def gradient_backprop(x,y,y_hat):
    #dl/dw
    return np.dot(2*x,y_hat-y).mean()


iters = 20
lr = 0.001

for epoch in range(iters):
    y_predicted = forward(x)
    l = loss(y_predicted,y)
    gradient = gradient_backprop(x,y,y_predicted)
    w -= lr * gradient
    print(f'epoch {epoch} ---> weights : {w} | loss : {l}')

predict = forward(7)
print(predict)