import torchvision
import torch

#加载网络模型，默认参数，无训练
vgg16  = torchvision.models.vgg16(pretrained=False)

#保存模型1:模型结构和参数
torch.sava(vgg16,"vgg16_method1.pth")

#加载模型1：加载的是结构
vgg16_load1 = torch.load("vgg16_method1.pth")

#---------------------------------------------------------
#---------------------------------------------------------
#---------------------------------------------------------
#---------------------------------------------------------

#保存模型2：模型参数(官方推荐，小一点)
torch.sava(vgg16.state_dict(),"vgg16_method2.pth")

#加载模型2：加载的是参数（权重）
vgg16_load2 = torch.load("vgg16_method2.pth")
#复原要先加载结构，再加载参数
vgg16  = torchvision.models.vgg16(pretrained=False)
vgg6_load=vgg16.load_state_dict(vgg16_load2)