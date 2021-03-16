import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

batch_size = 32
num_works = 6


class EPIDataset(Dataset):
    def __init__(self, name_idx):
        names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'all', 'all-NHEK']
        name = names[name_idx]
        Data_dir = '../data/%s/' % name
        enhancer = Data_dir + "/enhancer_of_%s.npy" % name
        promoter = Data_dir + "/promoter_of_%s.npy" % name
        label = Data_dir + "/label_of_%s.npy" % name
        self.en_data = np.load(enhancer)
        self.pr_data = np.load(promoter)
        self.x_data = [np.array(item)
                       for item in zip(self.en_data, self.pr_data)]
        self.label_data = np.load(label)
        self.len = len(self.label_data)

    def __getitem__(self, index):
        return self.x_data[index], self.label_data[index]

    def __len__(self):
        return self.len


epi_dataset = EPIDataset(4)
print("total:", epi_dataset.len)
np.set_printoptions(threshold=10000)  # 这个参数填的是你想要多少行显示
np.set_printoptions(linewidth=100)  # 这个参数填的是横向多宽
print("positive example:", np.sum(epi_dataset.label_data))
print("negative example:", epi_dataset.len - np.sum(epi_dataset.label_data))

generator = torch.Generator().manual_seed(116)
train_size = int(len(epi_dataset) * 0.9)
test_size = len(epi_dataset) - train_size
train_dataset, test_dataset = random_split(epi_dataset, [train_size, test_size], generator=generator)

print(train_dataset[0])
print(test_dataset[0])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_works)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_works)

print(len(train_loader))
print(len(test_loader))
