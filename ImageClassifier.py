# Check torch version and CUDA status if GPU is enabled.
import torch
print(torch.__version__)
print(torch.cuda.is_available()) # Should return True when GPU is enabled. 

# Imports here
import torch.nn.functional as F
import torch
from torch import nn, optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import numpy as np
import matplotlib
import sklearn

from collections import OrderedDict

import time

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Transforms that should be applied to the training, testing and validation sets
train_transforms = transforms.Compose([transforms.RandomRotation(15), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(), 
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

#loading each dataset from folder
train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)

test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)

valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_transforms)


#configuring each dataLoader with batchSize = 64
trainloader = DataLoader(train_dataset, batch_size = 64, shuffle = True, num_workers = 4)

testloader = DataLoader(test_dataset, batch_size = 64)

validloader = DataLoader(valid_dataset, batch_size = 64)


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#checking for cuda
device = ("cuda" if torch.cuda.is_available() else "cpu")

#importing VGG-16 model
model = models.vgg16(pretrained = True)

#freezing parameters in model
for param in model.parameters(): param.requires_grad = False


#redefining the classifier with hidden layer and 102 classes
classifier = nn.Sequential(nn.Linear(25088,7500), nn.ReLU(), nn.Dropout(0.3),
                           nn.Linear(7500,102), nn.LogSoftmax(dim=1))
                           
                          
#overweriting existing classifier
model.classifier = classifier

#only training classifier parameters
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.003)

#moving model to gpu if available
model.to(device)

epochs = 50
running_loss = 0
test_loss = 0
accuracy = 0
steps = 0

#starting at saved epoch from checkpoint
start_epoch = checkpoint['epoch']

print(device)

criterion = nn.NLLLoss()

#lists for matplotlib visualization
train_losses, test_losses = [], []

for epoch in range(start_epoch, start_epoch + epochs):
    #loading inputs and labels
    running_loss = 0
    steps += 1
    for inputs, labels in trainloader:
        #move inputs, labels to gpu
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad() #ressetting gradient descent
        
        logps = model.forward(inputs) #fowrard pass through network
        
        loss = criterion(logps, labels) #calculating loss
        
        #backwards and optimizer step
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    
    test_loss = 0
    accuracy = 0
    #only doing an evaluation run every 2nd iteration to save computation time
    if epoch % 2 == 0:
        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                log_ps = model(inputs)
                test_loss += criterion(log_ps, labels)
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                
    train_losses.append(running_loss/len(trainloader))
    test_losses.append(test_loss/len(testloader))
        
    model.train()
                
    print("Epoch: {}/{}.. ".format(epoch+1, epochs),
        "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
        "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
        "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
