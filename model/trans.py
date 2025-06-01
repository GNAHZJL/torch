import torchvision
from torch.utils.data import DataLoader
from model import Net
import torch
from torch.utils.tensorboard import SummaryWriter


#数据集
train_data = torchvision.datasets.CIFAR10("D:\\code\\python\\dataset",train=True,transform=torchvision.transforms.ToTensor(),download=False)

test_data = torchvision.datasets.CIFAR10("D:\\code\\python\\dataset",train=False,transform=torchvision.transforms.ToTensor(),download=False)

#数据长度
train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据{}".format(train_data_size))
print("测试数据{}".format(test_data_size))

#加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

#网络骨架
net=Net()

#损失函数
loss_fn = torch.nn.CrossEntropyLoss()

#优化器
learn_rate = 1e-2
optimizer = torch.optim.SGD(net.parameters(),lr=learn_rate )


# 添加tensorboard
writer = SummaryWriter("D:\\code\\python\\logs")

#训练网络
#训练次数
train_step=0
#测试次数
test_step=0
#轮数
epoch=1
for i in range(epoch):
    print("第{}轮训练开始".format(i+1))
    #net.train(),只对特定层有用，开不开无所谓，例如dropout就要开
    for data in train_dataloader:
        imgs,targets=data
        output=net.forward(imgs)
        loss=loss_fn(output,targets)

        #优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_step=train_step+1
        
        #直接输出loss是tensor类型数据，loss.item是类似int
        if train_step%100==0:
            print("训练{}次,loss:{}".format(train_step,loss))
            writer.add_scalar("tran_loss",loss.item(),train_step)

    #测试数据
    
    #net.eval(),只对特定层有用，开不开无所谓，例如dropout就要开
    total_test_loss=0
    total_accuracy=0
    #with = try .. except
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets=data
            output=net.forward(imgs)
            loss=loss_fn(output,targets)
            total_test_loss = total_test_loss+loss.item()
            #正确率算法
            accuracy=(output.argmax(1) == targets).sum()
            total_accuracy=total_accuracy+accuracy
    print("整体测试的loss:{}".format(total_test_loss))
    print("整体正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,test_step)
    test_step=test_step+1


    torch.save(net,"net_{}.pth".format(i))
    print("第{}模型已保存".format(i))
writer.close()
