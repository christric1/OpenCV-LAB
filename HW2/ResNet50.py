import cv2
import torch
import torch.nn as nn
import glob
from torch.utils.data import Dataset
import torch.utils.data as data
from random import choice
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models, datasets
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# hyperParameter
BATCH_SIZE = 16
EPOCHS = 5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data transformation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.Resize(128),
        transforms.ToTensor()
    ]),

    'train_argument': transforms.Compose([
        transforms.Resize(256),
        transforms.ColorJitter(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(128),
        transforms.ToTensor()
    ]),

    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
}

# Pass transforms in here, then run the next cell to see how the transforms look
dataset = datasets.ImageFolder('data/Q5_data/train', transform=data_transforms['train_argument'])

# Random split
train_set_size = int(len(dataset) * 0.8)
valid_set_size = len(dataset) - train_set_size
train_data, valid_data = data.random_split(dataset, [train_set_size, valid_set_size])

trainLoader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
validLoader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)


class ResNet50:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model = self.model.to(DEVICE)
        self.model.load_state_dict(torch.load('./model/ResNet50.pth', map_location='cpu'))

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
        img = cv2.imread("./chart/5.4_Accuracy.png")
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("img", img)


if __name__ == '__main__':

    print("device : ", DEVICE)

    # load model
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)
    summary(model, (3, 224, 224))

    writer = SummaryWriter(comment="ResNet50")

    # 設置訓練細節
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 參數
    itr = 1
    itr_ = 1
    p_itr = 200

    for epoch in range(EPOCHS):

        train_loss = 0.0

        # training
        model.train()
        print("Train : ")
        for batch_idx, Data in enumerate(trainLoader, 0):
            inputs, labels = Data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if itr % p_itr == 0:
                prediction = torch.argmax(outputs, dim=1)
                correct = prediction.eq(labels)
                acc = torch.mean(correct.float())

                print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.
                      format(epoch + 1, EPOCHS, itr, train_loss / p_itr, acc))

                writer.add_scalar("Loss/train", train_loss / p_itr, itr)
                writer.add_scalar("Accuracy/train", 100 * acc, itr)
                train_loss = 0.0

            itr += 1

        # valid
        # model.eval()
        # print("Valid : ")
        # for batch_idx, Data in enumerate(validLoader, 0):
        #
        #     inputs, labels = Data
        #     inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        #
        #     with torch.no_grad():
        #         outputs = model(inputs)
        #
        #         loss = criterion(outputs, labels)
        #         train_loss += loss.item()
        #
        #     if itr_ % p_itr == 0:
        #         prediction = torch.argmax(outputs, dim=1)
        #         correct = prediction.eq(labels)
        #         acc = torch.mean(correct.float())
        #
        #         print('[Epoch {}/{}] Iteration {} -> Valid Loss: {:.4f}, Accuracy: {:.3f}'.
        #               format(epoch + 1, EPOCHS, itr_, train_loss / p_itr, acc))
        #
        #         writer.add_scalar("Loss/valid", train_loss / p_itr, itr_)
        #         writer.add_scalar("Accuracy/valid", 100 * acc, itr_)
        #         train_loss = 0.0
        #
        #     itr_ += 1

    print('Finished Training')
    torch.save(model.state_dict(), 'model/ResNet50_argument.pth')  # save trained model
