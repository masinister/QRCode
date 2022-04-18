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
    y = enc(x)
    t_y = transform_encoded_img(y, device)
    out = dec(t_y)

    x = x.cpu().data.reshape((16,16))
    t_img = t_y.cpu().data.reshape((img_w,img_w))
    pred = out.round().int()

    fig, ax = plt.subplots(2)
    ax[0].imshow(x, interpolation='nearest', cmap = 'gray')
    ax[1].imshow(t_img, interpolation='nearest', cmap = 'gray')
    plt.show()


def transform_encoded_img(y, device):
    transform = torch.nn.Sequential(
        # T.RandomRotation(degrees=(0,10)),
        T.RandomPerspective(0.1,0.9),
    )
    img = transform(y)
    return img
