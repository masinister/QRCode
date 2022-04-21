import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import numpy as np

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc(x.view(x.size(0),64))
        return x.view(x.size(0),1,32,32)

    def save(self, path):
        torch.save(self.state_dict(), path)


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.fc(x.view(x.size(0),1024))
        return x.view(x.size(0),1,8,8)

    def save(self, path):
        torch.save(self.state_dict(), path)

if __name__ == '__main__':
    enc = Encoder()
    dec = Decoder()

    w = 16
    x = torch.zeros(w,w).unsqueeze(0).unsqueeze(0)

    print("Data shape:", x.shape)
    print("Encoded shape: ", enc(x).shape)
    print("Decoded shape: ", dec(enc(x)).shape)
