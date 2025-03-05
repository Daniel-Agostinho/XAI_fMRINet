# Third-party imports
import torch
from torch.utils.data import Dataset


class CNNDataset(Dataset):
    def __init__(self, data, labels, transform=True):

        if transform:
            self.data = torch.from_numpy(data).to(dtype=torch.float)
            self.labels = torch.from_numpy(labels).to(dtype=torch.float)

        else:
            self.data = data
            self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx, ...]
        labels = self.labels[idx, ...]
        return data, labels

    def __add__(self, other):
        temp_data = torch.cat((self.data, other.data), dim=0)
        temp_label = torch.cat((self.labels, other.labels), dim=0)
        return CNNDataset(temp_data, temp_label, transform=False)

    def set_device(self, device):
        self.data = self.data.to(device)
        self.labels = self.labels.to(device)


if __name__ == '__main__':
    pass
