import torch
from model import Net
from PIL import Image, ImageFilter, ImageDraw
import numpy as np

net = Net()
net.eval()
## 98.5%モデル
net.load_state_dict(torch.load('./trained_mnist_model', map_location=torch.device('cpu')))

def gaso(image):
    item = 0
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            item += image[x][y]

    item = item / (image.shape[0] * image.shape[1])
    #print(item)

    if item > 230:
        flag = 0
    else:
        flag = 1

    return flag

for i in range(12):
    filename = "./Result/test1_rank" + str(i+1) + "_item.jpeg"
    im = Image.open(filename)
    dd = 23
    image = []

    im1 = im.crop((65, 0, 65 + dd, 33))
    im1 = im1.resize((28, 28))
    im1 = np.array(im1)

    im2 = im.crop((65 - dd - 2, 0, 65 - 2, 33))
    im2 = im2.resize((28, 28))
    im2 = np.array(im2)

    im3 = im.crop((65 - 2 * dd - 4, 0, 65 - dd - 4, 33))
    im3 = im3.resize((28, 28))
    im3 = np.array(im3)

    flag = []

    flag.append(gaso(im1))
    flag.append(gaso(im2))
    flag.append(gaso(im3))


    for i in range(3):
        if flag[i] == 1:
            if i == 0:
                input = torch.tensor(im1).float()
                output = net(input)
                _, flag[i] = torch.max(output, 1)
            if i == 1:
                input = torch.tensor(im2).float()
                output = net(input)
                _, flag[i] = torch.max(output, 1)
            if i == 2:
                input = torch.tensor(im3).float()
                output = net(input)
                _, flag[i] = torch.max(output, 1)
        else:
            flag[2] = torch.tensor(flag[2])

    score = flag[2].item()*100 + flag[1].item()*10 + flag[0].item()
    print(score)

