import numpy as np
import torch
from torch.utils.data import DataLoader

from model import ConvEncoder, ConvDecoder
from dataset import TextDataset
from learning import train
from utils import train_test_split, testone

device = torch.device("cuda")
batch_size = 100
num_epochs = 10

text_length = 32
code_w = 32

data = TextDataset(length = text_length, device = device)

trainloader, testloader = train_test_split(data, 0.2, batch_size)

enc = ConvEncoder(input_size = text_length, code_w = code_w).to(device)

x = torch.zeros(text_length).unsqueeze(0).to(device)
img_w = enc(x).shape[-1]
print("Image size: ", img_w)

dec = ConvDecoder(input_size = (1, img_w, img_w), output_size = text_length).to(device)

enc, dec = train(enc, dec, img_w, trainloader, testloader, device, num_epochs)

x = (torch.cuda.FloatTensor(text_length).uniform_() > 0.5).float()

testone(enc, dec, x, img_w, device)
