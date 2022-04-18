import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import numpy as np

class ConvEncoder(nn.Module):

    def __init__(self):
        super(ConvEncoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),)

    def forward(self, x):
        x = self.conv(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)


class ConvDecoder(nn.Module):

    def __init__(self):
        super(ConvDecoder, self).__init__()

        self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2),                  
                nn.Conv2d(1, 128, 2, stride=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.2),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 2, stride=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.2),
                nn.Conv2d(64, 1, 2, stride=2),
                nn.Tanh(),)

    def forward(self, x):
        x = self.conv(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

if __name__ == '__main__':
    enc = ConvEncoder()
    dec = ConvDecoder()

    w = 16
    x = torch.zeros(w,w).unsqueeze(0).unsqueeze(0)

    print("Data shape:", x.shape)
    print("Encoded shape: ", enc(x).shape)
    print("Decoded shape: ", dec(enc(x)).shape)
