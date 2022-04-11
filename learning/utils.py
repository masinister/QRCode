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

def testone(enc, dec, x, shape, device):
    enc.eval()
    dec.eval()
    y = enc(x)
    out = dec(y)

    # img = y.cpu().data.reshape(shape)
    img = transform_encoded_img(y.unsqueeze(0), shape, device).cpu().data.reshape(shape)
    pred = out.round().int()
    plt.imshow(img, interpolation='nearest', cmap = 'gray')
    print(x)
    print(pred)
    print((pred == x).sum().item() / x.size(0))
    plt.show()


def transform_encoded_img(y, shape, device):
    img = y.view(y.size(0), shape[0], shape[1])
    transform = T.RandomPerspective(distortion_scale=0.4, p=1.0)
    img = transform(img)
    img = img.flatten(start_dim = 1)
    return img
