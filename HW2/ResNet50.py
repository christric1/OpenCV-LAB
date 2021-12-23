import cv2
import torch
import torch.nn as nn
import glob
from torch.utils.data import Dataset
from random import choice
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models, datasets
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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

    'train_random_erase': transforms.Compose([
        transforms.Resize((256, 256)),  # resize image to 224x224
        transforms.RandomResizedCrop(224),  # random crop and scale
        transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 標準化
        transforms.RandomErasing()
    ]),

    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
}

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder('data/Q5_data/train', transform=data_transforms['train_random_erase'])
trainLoader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)


class ResNet50:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model = self.model.to(DEVICE)
        self.model.load_state_dict(torch.load('./model/ResNet50.pth'))

    def Show_Model_Structure(self):
        summary(self.model, (3, 224, 224))

    def Show_Tensorboard(self):
        img1 = cv2.imread("./chart/loss.png")
        img2 = cv2.imread("./chart/accuracy.png")

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title("loss")
        plt.imshow(img1)

        plt.subplot(1, 2, 2)
        plt.title("accuracy")
        plt.imshow(img2)

        plt.show()

    def Test(self):
        images = glob.glob('.\\data\\Q5_data\\test\\*.jpg')
        img_path = choice(images)

        input_img = data_transforms['test'](Image.open(img_path))
        input_img = input_img.to(DEVICE)
        output = self.model(input_img.unsqueeze(0))

        m = nn.Softmax(dim=1)
        ratio = m(output)
        ratio = ratio.squeeze()
        ratio_np = ratio.cpu().detach().numpy()

        print("The ratio : ", ratio_np)

        plt.title("{}".format("Class:Cat" if np.argmax(ratio_np) == 0 else "Class:Dog"))
        plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        plt.show()

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
    torch.save(model.state_dict(), 'model/ResNet50_random_erase.pth')  # save trained model
