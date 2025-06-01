from PIL import Image
import torchvision
from model import Net
import torch

# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#导入数据
img_path = "model/img/image.png"  
img = Image.open(img_path)
img = img.convert("RGB")

#换数据，修改格式
transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()])
img = transform(img)

#转换成符合模型得数据，加batch_size
img = torch.reshape(img,(1,3,32,32))
img = img.to(device)


#加载模型
model = torch.load("model\\net_9.pth",map_location=device)

model.eval()
with torch.no_grad():
    output = model(img)
print(output.argmax(1))


