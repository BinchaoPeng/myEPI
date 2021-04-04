import numpy as np

name = "PBC"
feature_name = ""

trainPath = r'../data/epivan/%s/%s/%s_train.npz' % (name, feature_name, name)
# trainPath = r'../data/epivan/%s/%s_train.npz' % (name, name)
print(trainPath)
print(np.load(trainPath).files)
