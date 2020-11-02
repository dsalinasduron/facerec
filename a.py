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

class representation(nn.Module):
    def __init__(self):
        super(representation,self).__init__()
        self.c0 = nn.Conv2d(3,10,3, stride=2)
        self.c1 = nn.Conv2d(10,30,5)
        self.c2 = nn.Conv2d(30,10,3)
        self.fc3 = nn.Linear(11 * 11 * 10,20)
    def forward(self,x):
        x = f.max_pool2d( f.relu(self.c0(x)) ,kernel_size= 2)
        x = f.max_pool2d( f.relu(self.c1(x)) ,kernel_size= 2)
        x = f.max_pool2d( f.relu(self.c2(x)) ,kernel_size= 2)
        x = x.view(-1, 11 * 11 * 10)
        x = f.relu(self.fc3(x))
        return x

class head(nn.Module):
    def __init__(self):
        super(head,self).__init__()
        self.fc4 = nn.Linear(20,2)

    def forward(self,x):
        x = f.softmax(self.fc4(x),dim=1)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.r = representation()
        self.primaryHead = head()
        self.secondaryHead = head()
    def forward(self,x):
        x = self.r(x)
        x_primary = self.primaryHead(x)
        x_secondary = self.secondaryHead(x)
        return x_primary, x_secondary

def train(model,data,epochs=50):
    model = model.cuda()
    loss = f.cross_entropy

    op_r = torch.optim.Adam(model.r.parameters(),lr=0.0001)
    op_p = torch.optim.Adam(model.primaryHead.parameters(),lr=0.0001)
    op_s = torch.optim.Adam(model.secondaryHead.parameters(),lr=0.0001)

    for i in range(epochs):
        al = epoch(model,data,loss,op_r,op_s,op_p)
        print(al)

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

def epoch(model,data,loss,op_r,op_s,op_p):
    subset = trainingSize
    la = torch.tensor(0)

    # train secondary
    for p in model.r.parameters():
        p.requires_grad = False
    for i in range(10):
        break
        im, age, sex = data[i]
        im = im.cuda() # put in the gpu
        tg = torch.tensor([sex]).cuda() # make label a GPU tensor
        ot_p, ot_s = model(im)
        ls = loss(ot_s,tg)
        ls.backward()
        op.step()
        op.zero_grad()

    # train on primary loss and confusion loss
    # freeze secondary classfier
    for i in range(subset):
        im, age, sex = data[i]
        im = im.cuda() # put in the gpu
        tg = torch.tensor([age]).cuda() # make label a GPU tensor
        ot_p, ot_s = model(im)
        l_p = loss(ot_p,tg)
        print(torch.sum(torch.log(ot_s[0]) * -1))
        exit()
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
