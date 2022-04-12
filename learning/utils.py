import numpy as np
import torch
from torch import tensor
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
from matplotlib import pyplot as plt

def train_test_split(dataset, test_split, batch_size):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_size = int(test_split * dataset_size)
    test_idx = np.random.choice(indices, size=test_size, replace=False)
    train_idx = list(set(indices) - set(test_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = DataLoader(dataset, batch_size = batch_size, sampler=train_sampler)
    testloader = DataLoader(dataset, batch_size = batch_size, sampler=test_sampler)

    return trainloader, testloader

def testone(enc, dec, x, img_w, device):
    enc.eval()
    dec.eval()
    y = enc(x.unsqueeze(0))
    t_y = transform_encoded_img(y, device)
    out = dec(t_y)

    img = y.cpu().data.reshape((img_w,img_w))
    t_img = t_y.cpu().data.reshape((img_w,img_w))
    pred = out.round().int()

    fig, ax = plt.subplots(2)
    ax[0].imshow(img, interpolation='nearest', cmap = 'gray')
    ax[1].imshow(t_img, interpolation='nearest', cmap = 'gray')
    print(x)
    print(pred)
    print((pred == x).sum().item() / x.size(0))
    plt.show()


def transform_encoded_img(y, device):
    transform = torch.nn.Sequential(
        T.RandomPerspective(distortion_scale = 0.1, p = 0.9),
    )
    img = transform(y)
    return img
