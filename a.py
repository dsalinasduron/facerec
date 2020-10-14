import torchvision.datasets
import torch.nn as nn
import torch.nn.functional as f
import torch.optim
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
        print(age)
        im = self.tt(Image.open("../fairFace/" + filename))
        lw, up = age.split("-")
        age = int(lw) + int(up) / 2
        return im.unsqueeze(0), 1 if age < 30 else 0
    def setTest(self):
        self.list = [ i for i in open(self.val) ]

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
        x = f.softmax(self.fc4(x),dim=1)
        return x

def train(model,data,epochs=100):
    model.cuda()
    loss = f.cross_entropy
    op = torch.optim.SGD(model.parameters(),momentum=0.1,lr=0.01)
    for i in range(epochs):
        epoch(model,data,loss,op)

def test(model,data):
    subset = 10
    d = {0:0, 1:0}
    for i in range(subset):
        im, tg = data[i]
        im = im.cuda()
        tg = torch.tensor([tg]).cuda()
        ot = model(im)


def epoch(model,data,loss,op):
    subset = 10
    la = torch.tensor(0)
    for i in range(subset):
        im, tg = data[i]
        im = im.cuda()
        tg = torch.tensor([tg]).cuda()
        ot = model(im)
        ls = loss(ot,tg)
        la = la + ls
        la = la.detach()
        ls.backward()
        op.step()
        op.zero_grad()
    print(la/subset)

model = Net()
model = model.cuda()
data = Data()
#train(model,data)
data.setTest()
test(model,data)
