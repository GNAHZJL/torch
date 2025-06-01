from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

#记录运行过程日志，添加日志文件，
writer = SummaryWriter("D:\\code\\python\\logs")

#添加图表的标量，例如标题，x,y,全局变量等等
for i in range(100):
    writer.add_scalar("y=x",2*i,i)

img=Image.open("D:\\code\\python\\train\\ants\\0013035.jpg")
img_np=np.array(img)
writer.add_image("img",img_np,2,dataformats='HWC')

writer.close()