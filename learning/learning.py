import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from utils import transform_encoded_img

def train(enc, dec, img_w, trainloader, testloader, device, epochs):
    criterion = nn.MSELoss()
    params = list(enc.parameters()) + list(dec.parameters())
    optimizer = optim.Adam(params, lr = 1e-3)
    running_loss = 0.0


    for epoch in range(epochs):
        total = 0
        correct = 0
        enc.train()
        dec.train()
        bar = tqdm(enumerate(trainloader, 0), total = len(trainloader))
        for i, x in bar:
            X = Variable(x)

            optimizer.zero_grad()

            imgs = enc(X)
            imgs = transform_encoded_img(imgs, device)
            outputs = dec(imgs)
            pred = outputs.round().int()

            loss = criterion(outputs, X)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += X.size(0)


            correct += torch.all(X == pred, dim = 1).sum(0).item()
            accuracy = correct / total

            bar.set_description('Epoch %d, loss: %.3f, acc: %.3f'%(epoch + 1, running_loss, accuracy))

        test(enc, dec, img_w, testloader, device)
    return enc, dec

def test(enc, dec, img_w, dataloader, device):
    enc.eval()
    dec.eval()
    correct = 0
    total = 0
    bar = tqdm(enumerate(dataloader, 0), total = len(dataloader))
    with torch.no_grad():
        for i, x in bar:
            X = Variable(x)
            imgs = enc(X)
            imgs = transform_encoded_img(imgs, device)
            outputs = dec(imgs)

            pred = outputs.round().int()

            total += X.size(0)
            correct += torch.all(X == pred, dim = 1).sum(0).item()
            accuracy = correct / total
            bar.set_description('Accuracy: %.3f (%.3f / %.3f)' %(accuracy, correct, total))
