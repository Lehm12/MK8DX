# coding: UTF-8
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from torchvision.datasets import CIFAR10

import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from zero_out import zero_out
import numpy as np
import math

from model import Net

net = Net

transform_train = transforms.Compose([
        transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
transform_test = transforms.Compose([
        transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 64
# 訓練データとテストデータを用意
train_data = MNIST('~/tmp/mnist', train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = MNIST('~/tmp/mnist', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = Net()
net.to(device)

# cpuで実行する場合はコメントアウトする　てか133に組み込め
# with torch.cuda.device(0):
#   net.cuda()

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,25], gamma=0.1)

best_acc = 0

for epoch in range(30):
    running_loss = 0.0
    accuracy = 0.
    net.train()

    for i, (data, target) in enumerate(train_loader):
        inputs, labels = data.to(device), target.to(device)

        # 勾配情報をリセット
        optimizer.zero_grad()

        outputs = net(inputs)

        # コスト関数を使ってロスを計算する
        loss = criterion(outputs, labels)

        # 逆伝播
        loss.backward()

        # パラメータの更新
        optimizer.step()

        running_loss += loss.item()

        _, predict = torch.max(outputs, 1)
        c = (predict == labels).squeeze()

        accuracy += (torch.sum(c).cpu().float().data / batch_size).numpy()

        if i % 100 == 99:
            print('%d %d loss: %.3f accuracy: %.3f' % (epoch + 1, i + 1, running_loss / 100, accuracy / 100))
            running_loss = 0.0
            accuracy = 0.0

    scheduler.step()
    test_accuracy = 0.
    net.eval()
    for i, (data, target) in enumerate(test_loader):
        inputs, labels = data.to(device), target.to(device)
        # inputs, labels = data, target

        # Variableに変換
        inputs, labels = Variable(inputs), Variable(labels)

        outputs = net(inputs)

        _, predict = torch.max(outputs, 1)
        c = (predict == labels).squeeze()

        test_accuracy += (torch.sum(c).cpu().float().data / batch_size).numpy()
    print("test_accuracy :%.3f" % (test_accuracy / (i + 1)))
    if best_acc < (test_accuracy / (i + 1)):
        best_acc = (test_accuracy / (i + 1))
        torch.save(net.state_dict(), "./trained_mnist_model")
    print("best_accuray:%.3f" % best_acc)
print('Finished Training')
