import torchvision.datasets
import torch.nn as nn
import torch.nn.functional as f
from torchvision.transforms import ToTensor
from collections import defaultdict
from PIL import Image

class Data:
    def __init__(self):
        self.train = "../fairFace/fairface_label_train.csv"
        self.val = "../fairFace/fairface_label_val.csv"
        self.tt = ToTensor()
        """
        file,age,gender,race,service_test
        """
        self.list = [ i for i in open(self.train) ]
    def __getitem__(self,idx):
        filename,age,gender,race,service_test = self.list[idx+1].split(",")
        im = self.tt(Image.open("../fairFace/" + filename))
        return im.unsqueeze(0), 1 if gender == "Male" else 0

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.c0 = nn.Conv2d(3,10,5, stride=2)
        self.c1 = nn.Conv2d(10,10,5)
        self.c2 = nn.Conv2d(10,3,3)
        self.fc3 = nn.Linear(11 * 11 * 3,5)
        self.fc4 = nn.Linear(5,2)
    def forward(self,x):
        x = f.max_pool2d( f.relu(self.c0(x)) ,kernel_size= 2)
        x = f.max_pool2d( f.relu(self.c1(x)) ,kernel_size= 2)
        x = f.max_pool2d( f.relu(self.c2(x)) ,kernel_size= 2)
        x = x.view(-1, 11 * 11 * 3)
        x = f.relu(self.fc3(x))
        x = f.softmax(self.fc4(x))
        return x

model = Net()
data = Data()
print(model(data[0][0]))
