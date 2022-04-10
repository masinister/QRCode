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
        return 10000

if __name__ == '__main__':
    dataset = TextDataset()
    print(dataset[1234])
