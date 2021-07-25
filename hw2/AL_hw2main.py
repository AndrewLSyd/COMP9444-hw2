#!/usr/bin/env python3
"""
hw2main.py

UNSW COMP9444 Neural Networks and Deep Learning

DO NOT MODIFY THIS FILE
"""
import torch
import torchvision

from torch.utils.data import Dataset, random_split
from config import device

import student

# AL additions
import time
import gsheets
import datetime

import os
import shutil


# This class allows train/test split with different transforms
class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# Test network on validation set, if it exists.
def test_network(net,testloader):
    net.eval()
    total_images = 0
    total_correct = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    model_accuracy = total_correct / total_images * 100
    print('      Accuracy on {0} test images: {1:.2f}%'.format(
                                total_images, model_accuracy))
    net.train()
    
    return model_accuracy
    
def train_network(net,
                  criterion,
                  optimiser):
    print("Using device: {}"
          "\n".format(str(device)))
    ########################################################################
    #######                      Loading Data                        #######
    ########################################################################
    data = torchvision.datasets.ImageFolder(root=student.dataset)
    
    if student.train_val_split == 1:
        # Train on the entire dataset
        data = torchvision.datasets.ImageFolder(root=student.dataset,
                            transform=student.transform('train'))
        trainloader = torch.utils.data.DataLoader(data,
                            batch_size=student.batch_size, shuffle=True);
    else:
        # Split the dataset into trainset and testset
        data = torchvision.datasets.ImageFolder(root=student.dataset)
        data.len=len(data)
        train_len = int((student.train_val_split)*data.len)
        test_len = data.len - train_len
        train_subset, test_subset = random_split(data, [train_len, test_len])
        trainset = DatasetFromSubset(
            train_subset, transform=student.transform('train'))
        testset = DatasetFromSubset(
            test_subset, transform=student.transform('test'))

        trainloader = torch.utils.data.DataLoader(trainset, 
                            batch_size=student.batch_size, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, 
                            batch_size=student.batch_size, shuffle=False)

    ########################################################################
    #######                        Training                          #######
    ######################################################################## 
    now = datetime.datetime.now() + datetime.timedelta(hours=10)  # convert to Sydney time
    print("Start training at", now.strftime("%Y-%m-%d %H:%M:%S"))
    start_time = time.time()
    
    print(student.transform('train'))
    print(net)
    print(criterion)
    print(optimiser)
    
    source = 'student.py'
    target = 'models/student_' + now.strftime("%Y-%m-%d_%H%M") + '.py'

    shutil.copy(source, target)
    
    accuracy = []
    prev_valid_accuracy = 0
    
    for epoch in range(1,student.epochs+1):
        total_loss = 0
        total_images = 0
        total_correct = 0

        for batch in trainloader:           # Load batch
            images, labels = batch 

            images = images.to(device)
            labels = labels.to(device)
            


            preds = net(images)             # Process batch

#             print("DEBUG: images.shape", images.shape)
#             print("DEBUG: images.dtype", images.dtype)
            
#             print("DEBUG: labels.shape", labels.shape)
#             print("DEBUG: labels.dtype", labels.dtype)
#             print("DEBUG: labels", labels)
            
#             print("DEBUG: preds.shape", preds.shape)
#             print("DEBUG: preds.dtype", preds.dtype)
#             print("DEBUG: preds", preds)
            
            loss = criterion(preds, labels) # Calculate loss

            optimiser.zero_grad()
            loss.backward()                 # Calculate gradients
            optimiser.step()                # Update weights

            output = preds.argmax(dim=1)

            total_loss += loss.item()
            total_images += labels.size(0)
            total_correct += output.eq(labels).sum().item()

        model_accuracy = total_correct / total_images * 100
        print('epoch {0} total_correct: {1} loss: {2:.2f} acc: {3:.2f}'.format(
                   epoch,total_correct, total_loss, model_accuracy) )

        if epoch % 10 == 0:
            accuracy.append(model_accuracy)
            if student.train_val_split < 1:
                valid_accuracy = test_network(net, testloader)
                accuracy.append(valid_accuracy)
                
                
                
            torch.save(net.state_dict(), 'models/checkModel_' + now.strftime("%Y-%m-%d_%H%M") + '_epoch_' + str(epoch) + '.pth')
            print("      Model saved to checkModel.pth")
            
            # AL early stopping
            # if model does not improve by more than 1% after 10 epochs can it
            if (prev_valid_accuracy - valid_accuracy > 2) and epoch >= 1500:
                print("Earling stopping... prev_valid_accuracy", prev_valid_accuracy, "and valid_accuracy", valid_accuracy)
                break
                
            prev_valid_accuracy = valid_accuracy

    if student.train_val_split < 1:
        test_network(net,testloader)    
    torch.save(net.state_dict(), 'models/savedModel_' + now.strftime("%Y-%m-%d_%H%M") + '.pth')
    print("   Model saved to savedModel.pth")
    
    end_time = time.time()
    total_time = (end_time - start_time)
    
    # save results to google sheets
    # https://docs.google.com/spreadsheets/d/1x-BDCMig4xmOxJfoSZ0TUHQOJNpTFV1_YbyNiKxooWo/edit#gid=0
    result = [
        now.strftime("%Y-%m-%d_%H%M"),  # training_start
        total_time,  # run_time
        total_time / student.epochs,  # time per epoch
        str(student.transform('train')),  # transform train
        str(student.transform('test')),  # transform test
        str(net),  # network_structure
        str(optimiser),  # optimiser
        str(criterion),  # loss_function
        student.train_val_split,  # train_val_split
        student.batch_size,
        student.epochs,
        max(accuracy[::2]),  # max training accuracy
        max(accuracy[1::2]),  # max validation accuracy
    ]
    
    result += accuracy
    
    gsheets.write_to_gsheets(result)

    
def main():
    train_network(net=student.net.to(device),
                  criterion = student.lossFunc,
                  optimiser = student.optimiser)
    
if __name__ == '__main__':
    main()
