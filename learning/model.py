import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import numpy as np

class ConvEncoder(nn.Module):

    def __init__(self):
        super(ConvEncoder, self).__init__()

        self.conv = nn.Sequential(nn.ConvTranspose2d(1, 8, (2,2), (1,1)),
                                  nn.Dropout(0.02),
                                  # nn.ReLU(),
                                  nn.ConvTranspose2d(8, 8, (3,3), (2,2)),
                                  # nn.ReLU(),
                                  nn.ConvTranspose2d(8, 1, (4,4), (3,3)),
                                  nn.Dropout(0.02),
                                  # nn.ReLU(),
                                  )

    def forward(self, x):
        x = self.conv(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)


class ConvDecoder(nn.Module):

    def __init__(self):
        super(ConvDecoder, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(1, 8, (4,4), (3,3)),
                                  nn.Dropout(0.02),
                                  nn.ReLU(),
                                  nn.Conv2d(8, 8, (3,3), (2,2)),
                                  nn.Dropout(0.02),
                                  nn.ReLU(),
                                  nn.Conv2d(8, 1, (2,2), (1,1)),
                                  nn.Dropout(0.02),
                                  nn.ReLU(),
                                  )

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
    print("Decoded shape: ", enc(dec(x)).shape)
