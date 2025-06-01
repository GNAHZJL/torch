from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img=Image.open("D:\\code\\python\\train\\ants\\0013035.jpg")


#ToTensor
tran_img=transforms.ToTensor()
#等价于下面tran_img.__call__(img)
writer = SummaryWriter("D:\\code\\python\\logs")
writer.add_image("img",tran_img(img))


#Normalize 归一化,传入均值和标准差
trans_norm = transforms.Normalize([1,1,1],[0.5,0.5,0.5])
#然后归一化tensor格式类型
img_norm=trans_norm(tran_img(img))
writer.add_image("norm",img_norm)

#resize()
trans_resize = transforms.Resize((512,512))
img_resize=trans_resize(img)
writer.add_image("resize",tran_img(img_resize))
writer.close()

#compose

trans_resize_2 = transforms.Resize(512)
trans_compose=transforms.Compose([trans_resize_2,tran_img])
img_resize_2=trans_compose(img)
writer.add_image("resize",img_resize_2,1)