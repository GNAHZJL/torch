import torch
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("D:\\code\\python\\dataset",train=False,transform=torchvision.transforms.ToTensor(),download=False)

dataloader = DataLoader(dataset,batch_size=64)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.liner = torch.nn.Linear(196608,10)
    
    def forward(self,input):
        return self.liner(input)

net=Net()

for data in dataloader:
    imgs,targets = data
    output = torch.flatten(imgs)
    print(output.shape)
    output=net.forward(output)
    print(output.shape)