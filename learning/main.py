import numpy as np
import torch
from torch.utils.data import DataLoader

from model import Encoder, Decoder
from dataset import TextDataset
from learning import train
from utils import train_test_split, testone

device = torch.device("cuda")
text_length = 10
img_shape = (10,10)
img_size = np.prod(img_shape)
batch_size = 100
num_epochs = 2

data = TextDataset(length = text_length, device = device)

trainloader, testloader = train_test_split(data, 0.2, batch_size)

enc = Encoder(input_size = text_length, output_size = img_size).to(device)

dec = Decoder(input_size = img_size, output_size = text_length).to(device)

# print(enc(data[0].unsqueeze(0)).shape)

enc, dec = train(enc, dec, trainloader, testloader, device, num_epochs)

x = np.random.randint(2, size=text_length)

testone(enc, dec, x, img_shape, device)
