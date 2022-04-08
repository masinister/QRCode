import numpy as np
from torch import tensor
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
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

def testone(enc, dec, text, shape, device):
    y = enc(tensor(text, device = device).unsqueeze(0).float())
    img = y.cpu().data.reshape(shape)

    out = dec(y).cpu()
    pred = out.round().int().cpu().data[0].numpy()

    # res = out.clone().int().data.numpy()
    # res[out == 0] = 0
    # res[out != 0] = 1
    plt.imshow(img, interpolation='nearest', cmap = 'gray')
    print(text)
    print(pred)
    print((pred == text).sum() / len(text))
    plt.show()
