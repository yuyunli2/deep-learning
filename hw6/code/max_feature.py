import torch
import torchvision
import torchvision.transforms as transforms
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np
import time
import pickle

batch_size = 100
transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)

# Set up discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 196, 3, 1, 1)
        self.norm1 = nn.LayerNorm([196, 32, 32])

        self.conv2 = nn.Conv2d(196, 196, 3, 2, 1)
        self.norm2 = nn.LayerNorm([196, 16, 16])

        self.conv3 = nn.Conv2d(196, 196, 3, 1, 1)
        self.norm3 = nn.LayerNorm([196, 16, 16])
        self.conv4 = nn.Conv2d(196, 196, 3, 2, 1)
        self.norm4 = nn.LayerNorm([196, 8, 8])
        self.conv5 = nn.Conv2d(196, 196, 3, 1, 1)
        self.norm5 = nn.LayerNorm([196, 8, 8])
        self.conv6 = nn.Conv2d(196, 196, 3, 1, 1)
        self.norm6 = nn.LayerNorm([196, 8, 8])

        self.conv7 = nn.Conv2d(196, 196, 3, 1, 1)
        self.norm7 = nn.LayerNorm([196, 8, 8])
        self.conv8 = nn.Conv2d(196, 196, 3, 2, 1)
        self.norm8 = nn.LayerNorm([196, 4, 4])
        self.pool = nn.MaxPool2d(4, 4, 0)
        self.fc1  = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, 10)

    def forward(self, x, extract_features=0):
         x = F.leaky_relu(self.norm1(self.conv1(x)))  #1
#         assert(x.shape == torch.Size([batch_size, 196, 32, 32]))
         x = F.leaky_relu(self.norm2(self.conv2(x)))  #2
#         x = self.conv2(x)
#         x = F.leaky_relu(self.norm2(x))
#         assert(x.shape == torch.Size([batch_size, 196, 16, 16]))
         x = F.leaky_relu(self.norm3(self.conv3(x)))  #3
#         assert(x.shape == torch.Size([batch_size, 196, 16, 16]))
         x = F.leaky_relu(self.norm4(self.conv4(x)))  #4
         if(extract_features==4):
             h = F.max_pool2d(x, 8, 8)
             h = h.view(-1, 196)
             return h

#         assert(x.shape == torch.Size([batch_size, 196, 8, 8]))
         x = F.leaky_relu(self.norm5(self.conv5(x)))  #5
#         assert(x.shape == torch.Size([batch_size, 196, 8, 8]))
         x = F.leaky_relu(self.norm6(self.conv6(x)))  #6
#         assert(x.shape == torch.Size([batch_size, 196, 8, 8]))
         x = F.leaky_relu(self.norm7(self.conv7(x)))  #7
#         assert(x.shape == torch.Size([batch_size, 196, 8, 8]))
         x = F.leaky_relu(self.norm8(self.conv8(x)))  #8
         if(extract_features==8):
             h = F.max_pool2d(x, 4, 4)
             h = h.view(-1, 196)
             return h

#         assert(x.shape == torch.Size([batch_size, 196, 4, 4]))
         x = self.pool(x)
         x = x.view(batch_size, 196)

         f1 = self.fc1(x)
#         assert(f1.shape == torch.Size([batch_size, 1]))
#         assert(f10.shape == torch.Size([batch_size, 10]))
         return f1, f10

def plot(samples):
  #  print('samples', samples.shape)
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
 #       print('i, sample', i, sample)
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

model = torch.load('/u/training/tra411/scratch/tempD.model')
model.cuda()
model.eval()

batch_idx, (X_batch, Y_batch) = testloader.__next__()
X_batch = Variable(X_batch,requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()

X = X_batch.mean(dim=0)
X = X.repeat(batch_size,1,1,1)

Y = torch.arange(batch_size).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
for i in range(200):
    output = model(X, extract_features=8)

    loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/max_features_G8.png', bbox_inches='tight')
plt.close(fig)

for i in range(200):
    output = model(X, extract_features=4)

    loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/max_features_G4.png', bbox_inches='tight')
plt.close(fig)
