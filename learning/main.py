import numpy as np
import torch
from torch.utils.data import DataLoader

from model import Encoder, Decoder, ConvDecoder
from dataset import TextDataset
from learning import train
from utils import train_test_split, testone

device = torch.device("cuda")
text_length = 40
code_w = 20
img_w = code_w * 4
batch_size = 100
num_epochs = 10

data = TextDataset(length = text_length, device = device)

trainloader, testloader = train_test_split(data, 0.2, batch_size)

enc = Encoder(input_size = text_length, output_size = code_w * code_w).to(device)

dec = ConvDecoder(input_size = (1,img_w,img_w), output_size = text_length).to(device)

# print(enc(data[0].unsqueeze(0)).shape)

enc, dec = train(enc, dec, code_w, img_w, trainloader, testloader, device, num_epochs)

x = (torch.cuda.FloatTensor(text_length).uniform_() > 0.5).float()

testone(enc, dec, x, code_w, img_w, device)
