import torch
from torch.utils.tensorboard import SummaryWriter 

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #这里padding是根据网路计算出来的，3*32*32——>32*32*32;根据卷积公式，计算出来的
        # out = (in+2*padding - dila*(ker_size-1)-1)/stride + 1;默认dila=1，ker=5,stride=1,padding就等2 
        
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3,32,5,padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32,32,5,padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32,64,5,padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(1024,64),
            torch.nn.Linear(64,10)
        )
    
    def forward(self,x):
        x=self.model(x)
        return x
    
#实例化
net = Net()
#创建数据
input = torch.rand(64,3,32,32)
write = SummaryWriter("D:\\code\\python\\logs")
write.add_graph(net,input)
write.close()