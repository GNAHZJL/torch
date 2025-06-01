#网络骨架
import torch

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3,32,5,1,2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32,32,5,1,2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32,64,5,1,2),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(1024,64),
            torch.nn.Linear(64,10)
        )
    
    def forward(self,x):
        return self.model(x)

#测试网络
