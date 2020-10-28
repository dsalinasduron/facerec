import torchvision.datasets
import torch.nn as nn
import torch.nn.functional as f
import torch.optim
from torchvision.transforms import ToTensor
from collections import defaultdict
from PIL import Image
from itertools import islice

trainingSize = 100
ftrain = "../fairFace/fairface_label_train.csv"
fval = "../fairFace/fairface_label_val.csv"
class Data:
    def __init__(self, fname):
        """
        file,age,gender,race,service_test
        """
        self.tt = ToTensor()
        self.list = [ self.imageAgeGender(i) for i in islice(open(fname),1,trainingSize + 20) ]
        self.list  = [ i for i in self.list if i != None ]
    def __len__(self):
        return len(self.list)
    def __getitem__(self,idx):
        return self.list[idx]
    def imageAge(self,l):
        filename,age,gender,race,service_test = l.split(",")
        if "7" in age :
            age = 1
        else :
            lw, up = age.split("-")
            age = int(lw) + int(up) / 2
            age = 0 if age < 40 else 1
        try:
            im = self.tt(Image.open("../fairFace/" + filename))
        except :
            return None
        return im.unsqueeze(0), age
    def imageAgeGender(self,l):
        filename,age,gender,race,service_test = l.split(",")
        if gender == "Female" :
            gender = 0
        else :
            gender = 1
        if "7" in age :
            age = 1
        else :
            lw, up = age.split("-")
            age = int(lw) + int(up) / 2
            age = 0 if age < 40 else 1
        try:
            im = self.tt(Image.open("../fairFace/" + filename))
        except :
            return None
        return im.unsqueeze(0), age, gender

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.c0 = nn.Conv2d(3,10,3, stride=2)
        self.c1 = nn.Conv2d(10,30,5)
        self.c2 = nn.Conv2d(30,10,3)
        self.fc3 = nn.Linear(11 * 11 * 10,20)
        self.fc4 = nn.Linear(20,2)
        self.fc3_b = nn.Linear(11 * 11 * 10,20)
        self.fc4_b = nn.Linear(20,2)
    def forward(self,x):
        x = f.max_pool2d( f.relu(self.c0(x)) ,kernel_size= 2)
        x = f.max_pool2d( f.relu(self.c1(x)) ,kernel_size= 2)
        x = f.max_pool2d( f.relu(self.c2(x)) ,kernel_size= 2)
        x = x.view(-1, 11 * 11 * 10)
        x_a = f.relu(self.fc3(x))
        x_a = f.softmax(self.fc4(x_a),dim=1)
        x_b = f.relu(self.fc3_b(x))
        x_b = f.softmax(self.fc4_b(x_b),dim=1)
        return x_a, x_b

def train(model,data,epochs=50):
    model = model.cuda()
    loss = f.cross_entropy
    op = torch.optim.Adam(model.parameters(),lr=0.0001)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(op, mode='min', factor=0.1,
            patience=2, threshold=0.001, threshold_mode='rel', cooldown=0,
            min_lr=0, eps=1e-08, verbose=True)
    for i in range(epochs):
        al = epochAlg1(model,data,loss,op)
        print(al)
        sch.step(al)

def test(model,data):
    model = model.cuda()
    subset = 10
    d = {0:0, 1:0}
    for i in range(subset):
        im, tg = data[i]
        im = im.cuda()
        tg = torch.tensor([tg]).cuda()
        ot = model(im)
        print(ot,tg)

def epoch(model,data,loss,op):
    subset = trainingSize
    la = torch.tensor(0)
    for i in range(subset):
        im, tg = data[i]
        im = im.cuda()
        tg = torch.tensor([tg]).cuda()
        ot = model(im)
        ls = loss(ot,tg)
        ls.backward()
        op.step()
        op.zero_grad()
        la = la + ls
        la = la.detach()
    return (la/subset)

def epochAlg1(model,data,loss,op):
    subset = trainingSize
    la = torch.tensor(0)
    for i in range(subset):
        im, tg, tg_b = data[i]
        im = im.cuda()
        tg = torch.tensor([tg]).cuda()
        tg_b = torch.tensor([tg]).cuda()
        ot = model(im)

        for p in model.fc3_b.parameters() :
            p.requires_grad = False
        exit()
        ls = loss(ot,tg)
        ls.backward()
        op.step()
        op.zero_grad()
        la = la + ls
        la = la.detach()
    return (la/subset)


if __name__ == "__main__" :
    model = Net()
    trainingData = Data(ftrain)
    train(model,trainingData)
    #validationData = Data(fval)
    #test(model,validationData)
    #torch.save(model.state_dict(),"/pc/facerec/trained.pt")
