import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

nclasses = 20 

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()


        ## InceptionV3 feature extractor        
        self.inc = models.inception_v3(pretrained=True)
        self.inc.aux_logits = False
        # Freezing first layers
        for idx, item in self.inc.state_dict().items():
            item.requires_grad = False
        # Removing the softmax layer
        self.inc.fc = nn.Sequential()


        ## Resnext50 feature extractor        
        self.res = models.resnext50_32x4d(pretrained=True)

        for idx, item in self.res.state_dict().items():
        	item.requires_grad = False

        # Removing the softmax layer
        self.res.fc = nn.Sequential()

        ## Bottom of the NN
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.softmax = nn.Softmax()

        self.linear1 = nn.Linear(4096, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, nclasses)

    def forward(self, x):
        x1,x2 = self.res(x), self.inc(x).view(-1, 2048)
        x = torch.cat([x1,x2],1)
        x = self.linear1(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.ReLU(x)
        x = self.linear3(x)
        x = self.softmax(x)
		
        return x