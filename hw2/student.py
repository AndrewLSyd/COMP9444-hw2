#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.

Every single model has been saved
Tracking google sheets (see this public link) that documents all my experiments
Phase 1: building a baseline model
Phase 2: experimenting with transformers, holding model constant
Phase 3: experimenting 


"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """


    if mode == 'train':        
        my_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            
            
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.RandomHorizontalFlip(),

    #                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #                                        transforms.Resize([64, 64]),
    #                                        transforms.RandomCrop(60),
                                           transforms.ToTensor()
                                          ])

    elif mode == 'test':
        my_transform = transforms.Compose([
    #         transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.Grayscale(num_output_channels=1),
#             transforms.ColorJitter(brightness=.5, hue=.3),
#             transforms.RandomHorizontalFlip(),

    #                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #                                        transforms.Resize([64, 64]),
    #                                        transforms.RandomCrop(60),
                                           transforms.ToTensor()
                                          ])
    return my_transform

############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.number_hidden_units = 500
        self.flatten = nn.Flatten()  # take flatten method from super class
        
        self.model = nn.Sequential(
            nn.Linear(64*64, self.number_hidden_units),  # linear function, 28 by 28 inputs
            nn.Tanh(),
            nn.Linear(self.number_hidden_units, 14),  # 14 classes!
#             nn.ReLU()
#             ,
            nn.LogSoftmax(-1)  # log softmax to scale the 10 output classes
        )        
    def forward(self, x):
        x = self.flatten(x)  # flatten 2D input
        output = self.model(x)  # x (the input), feeds into the network self.linear_relu_stack and the output is returned
#         return torch.argmax(output, 1).long()
        return output


class CNN_best(nn.Module):
    def __init__(self):
        super(CNN_best, self).__init__()
        self.model = nn.Sequential(
            # need 14 outputs!
            # conv layer 1            
            nn.Conv2d(1, 64, kernel_size = 5, padding = 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.4),
            
            # conv layer 2
            nn.Conv2d(64, 128, kernel_size = 5, padding = 1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(2,2),            
            nn.Dropout(p=0.4),
            
            # conv layer 3
            nn.Conv2d(128, 256, kernel_size = 5, padding = 1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d(2,2),            
            nn.Dropout(p=0.4),
            
#              # conv layer 4
#             nn.Conv2d(256, 512, kernel_size = 5, padding = 1),
#             nn.BatchNorm2d(512),
#             nn.ELU(),
#             nn.MaxPool2d(2,2),            
#             nn.Dropout(p=0.4),
            
            # fully connected layer
            nn.Flatten(),
            nn.Linear(9216, 64),
            nn.ELU(),
            nn.Linear(64, 14),
            
            
            
#             nn.Dropout(p=0.5),  # dropout layer with dropout probability of 0.5
#             nn.ReLU(),
#             nn.Linear(64, 14),
#             nn.ReLU(),
#             nn.Softmax(-1)
        )        
    def forward(self, x):
#         x = self.flatten(x)  # flatten 2D input
        output = self.model(x)  # x (the input), feeds into the network self.linear_relu_stack and the output is returned
#         return torch.argmax(output, 1).long()
        return output


class CNN_testing_transformers(nn.Module):
    def __init__(self):
        super(CNN_testing_transformers, self).__init__()
        self.model = nn.Sequential(
            # need 14 outputs!
            # conv layer 1            
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 64, kernel_size = 5, padding = 1),            
            nn.ELU(),
            nn.MaxPool2d(2,2),            
            
            # conv layer 2
            nn.Conv2d(64, 128, kernel_size = 5, padding = 1),            
            nn.MaxPool2d(2,2),            
            nn.ELU(),                       
          
            
            # fully connected layer
            nn.Flatten(),
            nn.Linear(25088, 64),
            nn.ELU(),
            nn.Linear(64, 14),
            
            
            
#             nn.Dropout(p=0.5),  # dropout layer with dropout probability of 0.5
#             nn.ReLU(),
#             nn.Linear(64, 14),
#             nn.ReLU(),
#             nn.Softmax(-1)
        )        
    def forward(self, x):
#         x = self.flatten(x)  # flatten 2D input
        output = self.model(x)  # x (the input), feeds into the network self.linear_relu_stack and the output is returned
#         return torch.argmax(output, 1).long()
        return output

# class loss(nn.Module):
#     """
#     Class for creating a custom loss function, if desired.
#     If you instead specify a standard loss function,
#     you can remove or comment out this class.
#     """
#     def __init__(self):
#         super(loss, self).__init__()

#     def forward(self, output, target):
#         pass


# net = ANN()
net = CNN_best()

# net = CNN_testing_transformers()

# lossFunc = loss()
lossFunc = torch.nn.CrossEntropyLoss()  # pp187 says use logits as outputs and CrossEntropyLoss()
# lossFunc = torch.nn.CrossEntropyLoss
# lossFunc = torch.nn.NLLLoss
# lossFunc = torch.nn.functional.cross_entropy
# lossFunc = torch.nn.functional.nll_loss  # according to pp 181 of deep learning with pytorch
############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 256
epochs = 2000
# epochs = 10
# optimiser = optim.Adam(net.parameters(), lr=0.001)
optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True)