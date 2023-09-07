import numpy as np
import torch
y_pred = np.array([2.3,1.5,4.3])
y = np.array([0,0,1],dtype=np.float32)
def softmax(input):
    #exp(x1)/sum(exp(x1),exp(x2),...)
    result = []
    for i in input:
        fn = np.exp(i)/np.sum(np.exp(input))
        result.append(fn)
    return result
def categorical_cross_entropy(predicted,ground_truth):
    l = -np.sum(np.dot(ground_truth,np.log(predicted)))
    
    return l

softmax_outputs = softmax(y_pred)
print(softmax_outputs)
print(categorical_cross_entropy(softmax_outputs,y))

predicted_tensor = torch.from_numpy(y_pred).view(1,3)

ground_truth_tensor = torch.tensor([2])
torch_softmax = torch.softmax(predicted_tensor,dim=0)
print(torch_softmax)
torch_cross_entropy = torch.nn.CrossEntropyLoss() #softmax + cross entropy 

print('log :',np.log(0.8360188027814407))

print(torch_cross_entropy(predicted_tensor,ground_truth_tensor))