import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


batch_size = 128
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
#         print("x shape", x.shape)
         x = F.leaky_relu(self.norm1(self.conv1(x)))  #1
         assert(x.shape == torch.Size([128, 196, 32, 32]))
         x = F.leaky_relu(self.norm2(self.conv2(x)))  #2
         assert(x.shape == torch.Size([128, 196, 16, 16]))
         x = F.leaky_relu(self.norm3(self.conv3(x)))  #3
         assert(x.shape == torch.Size([128, 196, 16, 16]))
         x = F.leaky_relu(self.norm4(self.conv4(x)))  #4
         assert(x.shape == torch.Size([128, 196, 8, 8]))
         x = F.leaky_relu(self.norm5(self.conv5(x)))  #5
         assert(x.shape == torch.Size([128, 196, 8, 8]))
         x = F.leaky_relu(self.norm6(self.conv6(x)))  #6
         assert(x.shape == torch.Size([128, 196, 8, 8]))
         x = F.leaky_relu(self.norm7(self.conv7(x)))  #7
         assert(x.shape == torch.Size([128, 196, 8, 8]))
         x = F.leaky_relu(self.norm8(self.conv8(x)))  #8
         assert(x.shape == torch.Size([128, 196, 4, 4]))
         x = self.pool(x)
         x = x.view(batch_size, 196)
         
         f1 = self.fc1(x)
         assert(f1.shape == torch.Size([128, 1]))
         f10 = self.fc10(x)
         assert(f10.shape == torch.Size([128, 10]))
         return f1, f10

# Get into model
epochs = 100
learning_rate = 0.0001
model = discriminator()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


for epoch in range(0,epochs):
    if(epoch==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    if(epoch==75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/100.0


    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
    
        if(Y_train_batch.shape[0] < batch_size):
            continue
    
        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda()
        _, output = model(X_train_batch)
#        print("output", output)

        loss = criterion(output, Y_train_batch)
        optimizer.zero_grad()
    
        loss.backward()
        optimizer.step()

        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
    # Test on testloader
    if epoch % 10 == 0:
        correct = 0
        total = 0
        for batch_test_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
        
            if(Y_test_batch.shape[0] < batch_size):
                continue

            X_test_batch = Variable(X_test_batch).cuda()
            Y_test_batch = Variable(Y_test_batch).cuda()
            _test, output_test = model(X_test_batch)
#            print("output_test shape ", output_test.shape)
            _ori, predicted = torch.max(output_test.data, 1)
#            print("predicted shape", predicted.shape)
            correct += (predicted == Y_test_batch).sum().item()
#            print("Y_test_batch shape", Y_test_batch.shape)
            total += Y_test_batch.shape[0]
        accuracy = correct / total
        print(epoch, "accuracy on test is ",accuracy)


torch.save(model,'cifar10.model')
