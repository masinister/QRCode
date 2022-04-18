import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):

    def __init__(self, length = 128, device = torch.device("cuda")):
        self.device = device
        self.textlen = length

    def __getitem__(self, index):
        x = (torch.FloatTensor(self.textlen).uniform_() > torch.rand(1)).float()
        return x.to(self.device)

    def __len__(self):
        return 50000

class SquareDataset(Dataset):

    def __init__(self, width = 64, device = torch.device("cuda")):
        self.device = device
        self.width = width

    def __getitem__(self, index):
        x = torch.bernoulli(torch.FloatTensor(self.width, self.width).uniform_(0,1)).unsqueeze(0)
        return x.to(self.device)

    def __len__(self):
        return 50000

if __name__ == '__main__':
    dataset = SquareDataset()
    print(dataset[1234])
