import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

trans_sets=torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=dataset_transform,download=True)
test_sets=torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=dataset_transform,download=True)

writer = SummaryWriter("D:\\code\\python\\logs\\log")

for i in range(10):
    img,targer = test_sets[i]
    writer.add_image("img",img,i)

writer.close()