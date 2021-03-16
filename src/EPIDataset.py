from torch.utils.data import Dataset
import numpy as np

"""
load dataset
"""


class EPIDataset(Dataset):
    def __init__(self, name, is_train_set=True):
        trainPath = r'../data/epivan/%s/%s_train.npz' % (name, name)
        testPath = r'../data/epivan/%s/%s_test.npz' % (name, name)
        filename = trainPath if is_train_set else testPath
        data = np.load(filename)  # ['X_en_tra', 'X_pr_tra', 'y_tra'] / ['X_en_tes', 'X_pr_tes', 'y_tes']
        self.X_en = data[data.files[0]]
        self.X_pr = data[data.files[1]]
        self.X = [item for item in zip(self.X_en, self.X_pr)]
        # print(type(self.X))
        self.y = data[data.files[2]]
        self.len = len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len
