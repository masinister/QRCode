import numpy as np
import torch
from torch.utils.data import DataLoader

from model import Encoder, Decoder
from conv_model import ConvEncoder, ConvDecoder
from dataset import SquareDataset
from learning import train
from utils import train_test_split, testone

device = torch.device("cuda")
batch_size = 100
num_epochs = 10

code_w = 16

data = SquareDataset(width = code_w, device = device)

trainloader, testloader = train_test_split(data, 0.2, batch_size)

enc = Encoder().to(device)
dec = ConvDecoder().to(device)

x = data[0].unsqueeze(0)

print("Data shape:", x.shape)
print("Encoded shape: ", enc(x).shape)
print("Decoded shape: ", dec(enc(x)).shape)

img_w = enc(x).shape[-1]
testone(enc, dec, x, img_w, device)

enc, dec = train(enc, dec, trainloader, testloader, device, num_epochs)

testone(enc, dec, x, img_w, device)
