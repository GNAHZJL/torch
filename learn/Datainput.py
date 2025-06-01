#抽象类，要重写读取方法
from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        #拼合文件路径
        self.path = os.path.join(self.root_dir,self.label_dir)
        #获取文件夹下所有文件的名字，组成列表
        self.img_path=os.listdir(self.path)

     #返回图片信息和标签
    def __getitem__(self, index):
        #根据索引，获取文件名
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img=Image.open(img_item_path)
        label = self.label_dir
        return img,label
   
    def __len__(self):
        #文件夹文件多少
        return len(self.img_path)
    
Data_ants = MyData("D:\\code\\python\\train","ants")
Data_bees = MyData("D:\\code\\python\\train","bees")

#组合数据集，打开数据集
train_dataset=Data_ants+Data_bees
img,label = train_dataset[123]
img.show()

