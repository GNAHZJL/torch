import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#下载数据
test_tensor=torchvision.datasets.CIFAR10(root="D:\\code\\python\\dataset",train=False,transform=torchvision.transforms.ToTensor(),download=False)

#加载数据,每次取64张图
dataloader=DataLoader(test_tensor,batch_size=64)

#池化神经网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.max_pool = torch.nn.MaxPool2d(3,ceil_mode=True)
    
    def Max(self,input):
        output = self.max_pool(input)
        return output

#生成对象
net = Net()

#记录图像
write =  SummaryWriter("D:\\code\\python\\logs")


#执行代码
step = 0
for data in dataloader:
    imgs,target = data
    write.add_images("input",imgs,step)
    output = net.Max(imgs)
    write.add_images("output",output,step)
    step=step+1

write.close()