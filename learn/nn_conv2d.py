import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

#下载数据
test_tensor=torchvision.datasets.CIFAR10(root="D:\\code\\python\\dataset",train=False,transform=torchvision.transforms.ToTensor(),download=False)

#加载数据,每次取64张图
dataloader=DataLoader(test_tensor,batch_size=64)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #这里应该是设置这个神经网络的配置，3是3通道，生成6通道的，3*3的卷积核，1步长，0填充
        self.conv1 = nn.Conv2d(3,6,3,1,0)
    
    def forward(self,x):
        #使用这个网络，就是卷积了一下，返回
        x=self.conv1(x)
        return x

#生成对象
net = Net()

#记录图像
write =  SummaryWriter("D:\\code\\python\\logs")

#获取数据丢进神经网络中去
step=0
for data in dataloader:
    img,target = data
    output = net(img)
    #torch.size([64,3,32,32])
    write.add_images("ori",img,step)
    #torch.size([64,6,30,30])
    output=torch.reshape(output,(-1,3,30,30))
    write.add_images("conv",output,step)
    step=step+1

write.close()