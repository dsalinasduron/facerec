import torchvision.datasets
import torch.nn as nn
from torchvision.transforms import ToTensor
from collections import defaultdict

faces = torchvision.datasets.CelebA("/pc",transform=ToTensor(),target_type="identity",target_transform=int)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

model = Net()

d = defaultdict(lambda : 0)
for i in range(len(faces)) :
    im, tg = faces[i]
    d[tg] = d[tg] + 1

sortedKeys = sorted(d, key=lambda k : -1 * d[k])
for i in range(10):
    print(sortedKeys[i], d[sortedKeys[i]])
