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

batch_size = 100
# Transform data and get into dataloader
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
print("Successfully load data!")

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

    def forward(self, x):
#         print('x d1 shape', x)
         x = F.leaky_relu(self.norm1(self.conv1(x)))  #1
#         print('after x1 shape', x)
         assert(x.shape == torch.Size([batch_size, 196, 32, 32]))
         x = F.leaky_relu(self.norm2(self.conv2(x)))  #2
         assert(x.shape == torch.Size([batch_size, 196, 16, 16]))
         x = F.leaky_relu(self.norm3(self.conv3(x)))  #3
         assert(x.shape == torch.Size([batch_size, 196, 16, 16]))
         x = F.leaky_relu(self.norm4(self.conv4(x)))  #4
         assert(x.shape == torch.Size([batch_size, 196, 8, 8]))
         x = F.leaky_relu(self.norm5(self.conv5(x)))  #5
         assert(x.shape == torch.Size([batch_size, 196, 8, 8]))
         x = F.leaky_relu(self.norm6(self.conv6(x)))  #6
         assert(x.shape == torch.Size([batch_size, 196, 8, 8]))
         x = F.leaky_relu(self.norm7(self.conv7(x)))  #7
         assert(x.shape == torch.Size([batch_size, 196, 8, 8]))
         x = F.leaky_relu(self.norm8(self.conv8(x)))  #8
         assert(x.shape == torch.Size([batch_size, 196, 4, 4]))
         x = self.pool(x)
         x = x.view(batch_size, 196)

         f1 = self.fc1(x)
         assert(f1.shape == torch.Size([batch_size, 1]))
         f10 = self.fc10(x)
         assert(f10.shape == torch.Size([batch_size, 10]))
         return f1, f10

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(100, 196*4*4)
        self.normfc = nn.BatchNorm1d(196*4*4)
        self.conv1 = nn.ConvTranspose2d(196, 196, 4, 2, 1) #1
        self.norm1 = nn.BatchNorm2d(196)         

        self.conv2 = nn.Conv2d(196, 196, 3, 1, 1)          #2
        self.norm2 = nn.BatchNorm2d(196)

        self.conv3 = nn.Conv2d(196, 196, 3, 1, 1)          #3
        self.norm3 = nn.BatchNorm2d(196)
        self.conv4 = nn.Conv2d(196, 196, 3, 1, 1)          #4
        self.norm4 = nn.BatchNorm2d(196)
        self.conv5 = nn.ConvTranspose2d(196, 196, 4, 2, 1) #5
        self.norm5 = nn.BatchNorm2d(196)
        self.conv6 = nn.Conv2d(196, 196, 3, 1, 1)          #6
        self.norm6 = nn.BatchNorm2d(196)
        self.conv7 = nn.ConvTranspose2d(196, 196, 4, 2, 1) #7
        self.norm7 = nn.BatchNorm2d(196)        
        self.conv8 = nn.Conv2d(196, 3, 3, 1, 1)            #8

    def forward(self, x):
#         print('x before fc', x.shape)
         x = self.normfc(self.fc1(x))  #fc
 #        print('x after fc', x.shape)
         assert(x.shape == torch.Size([batch_size, 196*4*4]))
         x = x.view(batch_size, 196, 4, 4)
#         x = F.relu(self.norm1(self.conv1(x)))  #1
         x = self.conv1(x)
#         print('x conv1', x.shape)
         x = self.norm1(x)
#         print('x norm1', x.shape)
         x = F.relu(x) 
#         print('x relu', x.shape)
         assert(x.shape == torch.Size([batch_size, 196, 8, 8]))
         x = F.relu(self.norm2(self.conv2(x)))  #2
         assert(x.shape == torch.Size([batch_size, 196, 8, 8]))
         x = F.relu(self.norm3(self.conv3(x)))  #3
         assert(x.shape == torch.Size([batch_size, 196, 8, 8]))
         x = F.relu(self.norm4(self.conv4(x)))  #4
         assert(x.shape == torch.Size([batch_size, 196, 8, 8]))
         x = F.relu(self.norm5(self.conv5(x)))  #5
         assert(x.shape == torch.Size([batch_size, 196, 16, 16]))
         x = F.relu(self.norm6(self.conv6(x)))  #6
         assert(x.shape == torch.Size([batch_size, 196, 16, 16]))
         x = F.relu(self.norm7(self.conv7(x)))  #7
         assert(x.shape == torch.Size([batch_size, 196, 32, 32]))
         x = F.tanh(self.conv8(x))  #8
         assert(x.shape == torch.Size([batch_size, 3, 32, 32]))


         return x








def calc_gradient_penalty(netD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    alpha = alpha.cuda()
    
    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


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




# Create two networks and an optimizer for each
aD =  discriminator()
aD.cuda()

aG = generator()
aG.cuda()

optimizer_g = torch.optim.Adam(aG.parameters(), lr=0.0001, betas=(0,0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=0.0001, betas=(0,0.9))

criterion = nn.CrossEntropyLoss()

# Random batch noise for the generator
gen_train = 1
num_epochs = 200 
n_z = 100
n_classes = 10
np.random.seed(352)
label = np.asarray(list(range(10))*10)
noise = np.random.normal(0,1,(batch_size,n_z))
label_onehot = np.zeros((batch_size,n_classes))
#print('label', label.shape, label)
#print('label_onehot', label_onehot.shape, label_onehot)
label_onehot[np.arange(batch_size), label] = 1
noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
noise = noise.astype(np.float32)

save_noise = torch.from_numpy(noise)
save_noise = Variable(save_noise).cuda()




start_time = time.time()

# Train the model
for epoch in range(0,num_epochs):

    # before epoch training loop starts
    loss1 = []
    loss2 = []
    loss3 = []
    loss4 = []
    loss5 = []
    acc1 = []

    aG.train()
    aD.train()
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

        if(Y_train_batch.shape[0] < batch_size):
            continue

        # train G
        if((batch_idx%gen_train)==0):
            for p in aD.parameters():
                p.requires_grad_(False)
        
            aG.zero_grad()
        
            label = np.random.randint(0,n_classes,batch_size)
            noise = np.random.normal(0,1,(batch_size,n_z))
            label_onehot = np.zeros((batch_size,n_classes))
            label_onehot[np.arange(batch_size), label] = 1
            noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
            noise = noise.astype(np.float32)
            noise = torch.from_numpy(noise)
            noise = Variable(noise).cuda()
            fake_label = Variable(torch.from_numpy(label)).cuda()
        
            fake_data = aG(noise)
            gen_source, gen_class  = aD(fake_data)
        
            gen_source = gen_source.mean()
            gen_class = criterion(gen_class, fake_label)
        
            gen_cost = -gen_source + gen_class
            gen_cost.backward()
        
            optimizer_g.step()
    
            for group in optimizer_g.param_groups:
                for p in group['params']:
                    state = optimizer_g.state[p]
                    if('step' in state and state['step']>=1024):
                        state['step'] = 1000
    
        # train D
        for p in aD.parameters():
            p.requires_grad_(True)
        
        aD.zero_grad()
        
        # train discriminator with input from generator
        label = np.random.randint(0,n_classes,batch_size)
        noise = np.random.normal(0,1,(batch_size,n_z))
        label_onehot = np.zeros((batch_size,n_classes))
        label_onehot[np.arange(batch_size), label] = 1
        noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
        noise = noise.astype(np.float32)
        noise = torch.from_numpy(noise)
        noise = Variable(noise).cuda()
        fake_label = Variable(torch.from_numpy(label)).cuda()
        with torch.no_grad():
            fake_data = aG(noise)
        
        disc_fake_source, disc_fake_class = aD(fake_data)
        
        disc_fake_source = disc_fake_source.mean()
        disc_fake_class = criterion(disc_fake_class, fake_label)
        
        # train discriminator with input from the discriminator
        real_data = Variable(X_train_batch).cuda()
        real_label = Variable(Y_train_batch).cuda()
        
        disc_real_source, disc_real_class = aD(real_data)
        
        prediction = disc_real_class.data.max(1)[1]
        accuracy = ( float( prediction.eq(real_label.data).sum() ) /float(batch_size))*100.0
        
        disc_real_source = disc_real_source.mean()
        disc_real_class = criterion(disc_real_class, real_label)
        
        gradient_penalty = calc_gradient_penalty(aD,real_data,fake_data)
        
        # disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
        disc_cost = disc_fake_source - disc_real_source + gradient_penalty + disc_real_class
        disc_cost.backward()
        
        optimizer_d.step()
        for group in optimizer_d.param_groups:
            for p in group['params']:
                state = optimizer_d.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000

        disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty

        # within the training loop
        loss1.append(gradient_penalty.item())
        loss2.append(disc_fake_source.item())
        loss3.append(disc_real_source.item())
        loss4.append(disc_real_class.item())
        loss5.append(disc_fake_class.item())
        acc1.append(accuracy)
        if((batch_idx%50)==0):
            print(epoch, batch_idx, "%.2f" % np.mean(loss1), 
                                    "%.2f" % np.mean(loss2), 
                                    "%.2f" % np.mean(loss3), 
                                    "%.2f" % np.mean(loss4), 
                                    "%.2f" % np.mean(loss5), 
                                    "%.2f" % np.mean(acc1))

    # Test the model
    aD.eval()
    with torch.no_grad():
        test_accu = []
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
            if(Y_test_batch.shape[0] < batch_size):
                continue
            X_test_batch, Y_test_batch= Variable(X_test_batch).cuda(),Variable(Y_test_batch).cuda()
    
            with torch.no_grad():
                _, output = aD(X_test_batch)
    
            prediction = output.data.max(1)[1] # first column has actual prob.
            accuracy = ( float( prediction.eq(Y_test_batch.data).sum() ) /float(batch_size))*100.0
            test_accu.append(accuracy)
            accuracy_test = np.mean(test_accu)
    print('Testing',accuracy_test, time.time()-start_time)
    
    ### save output
    with torch.no_grad():
        aG.eval()
        samples = aG(save_noise)
        samples = samples.data.cpu().numpy()
        samples += 1.0
        samples /= 2.0
        samples = samples.transpose(0,2,3,1)
        aG.train()
    
    if(epoch%10 == 9):
        fig = plot(samples)
        plt.savefig('output/%s.png' % str(epoch).zfill(3), bbox_inches='tight')
        plt.close(fig)
    
    if(((epoch+1)%1)==0):
        torch.save(aG,'tempG.model')
        torch.save(aD,'tempD.model')

torch.save(aG,'generator.model')
torch.save(aD,'discriminator.model')

