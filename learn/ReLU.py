import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("D:\\code\\python\\dataset",train=False,transform=torchvision.transforms.ToTensor(),download=False)

dataloader = DataLoader(dataset,batch_size=64)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.ReLU = torch.nn.ReLU()
        self.sigmod = torch.nn.Sigmoid()
    
    def relu(self,input):
        return self.ReLU(input)
        
    
    def sigmoid(self,input):
        return self.sigmod(input)
       

net = Net()
write =  SummaryWriter("D:\\code\\python\\logs")
step=0
for data in dataloader:
    imgs,targets = data
    output1=net.relu(imgs)
    write.add_images("output1",output1,step)
    output2=net.sigmod(imgs)
    write.add_images("output2",output2,step)
    step=step+1

write.close()