import numpy as np
import torch
ground_truth = torch.tensor([0,0,1],dtype=torch.float32)
predicted_outputs = torch.tensor([0.3,2.5,4.3],dtype=torch.float32)

def softmax(input):
    output = []
    for i in input:
       output.append(np.exp(i)/np.sum(np.exp(input)))
    return output

def crossEntropy(groundtruth,outputs):
    
   return -np.sum(np.dot(groundtruth,np.log(outputs)))

#tensor = torch.from_numpy(predicted_outputs)

#softmax_ = torch.softmax(tensor,dim=0)

crossEntropy_ = torch.nn.CrossEntropyLoss()
print(crossEntropy_(predicted_outputs,ground_truth))
