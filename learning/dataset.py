import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):

    def __init__(self, length = 128, device = torch.device("cuda")):
        self.device = device
        self.textlen = length

    def __getitem__(self, index):
        mask = 2**torch.arange(self.textlen).to(self.device)
        return torch.tensor(index, device = self.device).unsqueeze(-1).bitwise_and(mask).ne(0).float()

    def __len__(self):
        return 10000

if __name__ == '__main__':
    dataset = TextDataset()
    print(dataset[12345])
