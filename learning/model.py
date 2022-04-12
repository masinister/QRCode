import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import numpy as np

class Encoder(nn.Module):

    def __init__(self, input_size, output_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc = nn.Sequential(nn.Linear(self.input_size, 400),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(400, self.output_size),
                                nn.Dropout(0.1),
                                nn.ReLU())

    def forward(self, x):
        x = self.fc(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

class Decoder(nn.Module):

    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc = nn.Sequential(nn.Linear(self.input_size, 400),
                                  nn.Dropout(0.1),
                                  nn.ReLU(),
                                  nn.Linear(400, self.output_size),
                                  nn.Dropout(0.1),
                                  nn.ReLU())

    def forward(self, x):
        x = self.fc(x)
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
    enc = Encoder(100,(20,20))
    print(enc)
