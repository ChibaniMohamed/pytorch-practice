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

predicted_tensor = torch.from_numpy(np.array([0.7,0.2,0.1]))
ground_truth_tensor = torch.from_numpy(np.array([1,0,0],dtype=np.float32))
torch_softmax = torch.softmax(predicted_tensor,dim=0)
torch_cross_entropy = torch.binary_cross_entropy_with_logits(torch_softmax,ground_truth_tensor)
print(torch_cross_entropy)