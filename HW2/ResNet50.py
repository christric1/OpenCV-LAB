import cv2
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch import optim
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from torchvision import datasets
from torchsummary import summary
from tqdm import tqdm
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import *
from PIL import Image
import os

# hyperParameter
BATCH_SIZE = 20
EPOCHS = 20
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data transformation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),  # resize image to 224x224
        transforms.RandomResizedCrop(224),  # random crop and scale
        transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 標準化
    ]),

    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
}

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder('data/Q5_data/train', transform=data_transforms['train'])

# # Random split
# train_set_size = int(len(train_data) * 0.8)
# valid_set_size = len(train_data) - train_set_size
# train_set, valid_set = data.random_split(train_data, [train_set_size, valid_set_size])

trainLoader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
# validLoader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)


class ResNet50:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model = self.model.to(DEVICE)

    def Show_Model_Structure(self):
        summary(self.model, (3, 224, 224))

    def Show_Tensorboard(self):
        print("123")

    def Test(self):
        print("123")

    def Data_Argument(self):
        print("123")


if __name__ == '__main__':

    print("device : ", DEVICE)

    # load model
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)
    summary(model, (3, 224, 224))

    writer = SummaryWriter(comment="ResNet50")

    # 設置訓練細節
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # training
    for epoch in range(EPOCHS):
        # train_Loss, train_Acc = train(model, trainLoader, optimizer, criterion, epoch)

        train_loss = 0.0
        train_acc = 0.0
        percent = 10

        for batch_idx, Data in enumerate(trainLoader, 0):
            inputs, labels = Data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            prediction = outputs.max(1, keepdim=True)[1]
            acc = prediction.eq(labels.view_as(prediction)).sum().item()
            train_acc += acc

            if batch_idx % percent == 0:
                print('train epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}\t'.format(
                    epoch, (batch_idx + 1) * len(inputs), len(train_data),
                           100.0 * batch_idx / len(trainLoader), loss))

        train_loss *= BATCH_SIZE
        train_loss /= len(train_data)
        train_acc = train_acc / len(train_data)

        print('\ntrain epoch: {}\tloss: {:.6f}\taccuracy:{:.4f}% '.format(epoch, train_loss, 100. * train_acc))

        writer.add_scalar("training loss ", train_loss, epoch)
        writer.add_scalar("accuracy", 100 * train_acc, epoch)

    print('Finished Training')
    torch.save(model.state_dict(), '../model/ResNet50.pth')  # save trained model

