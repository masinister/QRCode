import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import numpy as np

class ConvEncoder(nn.Module):

    def __init__(self, input_size, code_w):
        super(ConvEncoder, self).__init__()
        self.input_size = input_size
        self.code_w = code_w
        self.code_size = self.code_w ** 2

        self.fc = nn.Sequential(nn.Linear(self.input_size, self.code_size),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(self.code_size, self.code_size),
                                nn.Dropout(0.1),
                                nn.ReLU())

        self.fc_size = self.fc(torch.zeros(input_size).unsqueeze(0)).shape

        self.conv = nn.Sequential(nn.ConvTranspose2d(1, 4, (2,2), (2,2)),
                                  nn.ConvTranspose2d(4, 8, (2,2), (2,2)),
                                  nn.ConvTranspose2d(8, 16, (2,2), (2,2)),
                                  nn.Conv2d(16,1,(2,2),(2,2)))

    def forward(self, x):
        x = self.fc(x).view((x.size(0), 1, self.code_w, self.code_w))
        x = self.conv(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)


class ConvDecoder(nn.Module):

    def __init__(self, input_size, output_size):
        super(ConvDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.conv = nn.Sequential(nn.Conv2d(1, 16, (4,4), (4,4)),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Conv2d(16, 32, (3,3), (2,2)),
                                nn.Dropout(0.1),
                                nn.ReLU(),)

        self.fc_size = self.conv(torch.zeros(input_size).unsqueeze(0)).flatten().shape[-1]
        # print(self.fc_size)

        self.fc = nn.Sequential(nn.Linear(self.fc_size, 400),
                                  nn.Dropout(0.1),
                                  nn.ReLU(),
                                  nn.Linear(400, self.output_size),
                                  nn.Dropout(0.1),
                                  nn.ReLU())

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

if __name__ == '__main__':
    enc = ConvEncoder(10, 32)
    x = torch.zeros(10).unsqueeze(0)
    print(enc(x).shape)
