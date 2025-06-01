from torch import nn
import torch
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__() 

    def forward(self,input):
        output = input+1
        return output
    

x=torch.tensor([[1,0,1],
                [0,1,0],
                [1  ,0,2]])
y=torch.tensor([[1,2],
                [2,3],
                ])
x=torch.reshape(x,(1,1,3,3))
y=torch.reshape(y,(1,1,2,2))
print(F.conv2d(x,y,stride=1))