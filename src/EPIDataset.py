from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader

"""
load dataset
"""


class EPIDataset(Dataset):
    """
    关于EPI路径的问题：

    采用epivan的基本路径，对于最后的npz文件，增加一个父目录表示不同的方式提取到的特征,feature_name
    """

    def __init__(self, cell_name, feature_name="", is_train_set=True):
        trainPath = r'../data/epivan/%s/features/%s/%s_train.npz' % (cell_name, feature_name, cell_name)
        testPath = r'../data/epivan/%s/features/%s/%s_test.npz' % (cell_name, feature_name, cell_name)
        filename = trainPath if is_train_set else testPath
        data = np.load(filename)  # ['X_en_tra', 'X_pr_tra', 'y_tra'] / ['X_en_tes', 'X_pr_tes', 'y_tes']
        if feature_name.find("longformer") >= 0 or feature_name.find("elmo") >= 0:
            print("feature_name:", feature_name)
            self.X = data[data.files[0]]
            self.y = data[data.files[1]]
        else:
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


if __name__ == '__main__':
    trainSet = EPIDataset("pbc_IMR90", feature_name="longformer-hug")
    len_trainSet = len(trainSet)
    print("trainSet data len:", len(trainSet))
    trainLoader = DataLoader(dataset=trainSet, batch_size=4, shuffle=True, num_workers=0)
    print("trainLoader len:", len(trainLoader))
    for id, (x, y) in enumerate(trainLoader, 0):
        print(id)
        print("X:")
        print(x)
        print("Y:")
        print(y)
        if id == 2:
            break
