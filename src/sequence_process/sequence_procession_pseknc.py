import numpy as np
from dataUtils.pseknc.PseKNC_II import PseKNC
import os

# In[]:
names = ['pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
cell_name = names[4]
feature_name = "pseknc"

train_dir = '../../data/epivan/%s/train/' % cell_name
imbltrain = '../../data/epivan/%s/imbltrain/' % cell_name
test_dir = '../../data/epivan/%s/test/' % cell_name
Data_dir = '../../data/epivan/%s/%s/' % (cell_name, feature_name)

if os.path.exists(Data_dir):
    print("path exits!")
else:
    os.mkdir(Data_dir)
    print("path created!")

print('Experiment on %s dataset' % cell_name)

print('Loading seq data...')
enhancers_tra = open(train_dir + '%s_enhancer.fasta' % cell_name, 'r').read().splitlines()[1::2]
promoters_tra = open(train_dir + '%s_promoter.fasta' % cell_name, 'r').read().splitlines()[1::2]
y_tra = np.loadtxt(train_dir + '%s_label.txt' % cell_name)

im_enhancers_tra = open(imbltrain + '%s_enhancer.fasta' % cell_name, 'r').read().splitlines()[1::2]
im_promoters_tra = open(imbltrain + '%s_promoter.fasta' % cell_name, 'r').read().splitlines()[1::2]
y_imtra = np.loadtxt(imbltrain + '%s_label.txt' % cell_name)

enhancers_tes = open(test_dir + '%s_enhancer_test.fasta' % cell_name, 'r').read().splitlines()[1::2]
promoters_tes = open(test_dir + '%s_promoter_test.fasta' % cell_name, 'r').read().splitlines()[1::2]
y_tes = np.loadtxt(test_dir + '%s_label_test.txt' % cell_name)

print('平衡训练集')
print('pos_samples:' + str(int(sum(y_tra))))
print('neg_samples:' + str(len(y_tra) - int(sum(y_tra))))
print('不平衡训练集')
print('pos_samples:' + str(int(sum(y_imtra))))
print('neg_samples:' + str(len(y_imtra) - int(sum(y_imtra))))
print('测试集')
print('pos_samples:' + str(int(sum(y_tes))))
print('neg_samples:' + str(len(y_tes) - int(sum(y_tes))))


# In[ ]:
def get_data(enhancers, promoters):
    X_en1, X_pr1, X_en2, X_pr2 = [], [], [], []
    for en, pr in zip(enhancers, promoters):
        pseknc = PseKNC(seq=en)
        en1, en2 = pseknc()
        pseknc = PseKNC(seq=pr)
        pr1, pr2 = pseknc()
        X_en1.append(en1)
        X_pr1.append(pr1)
        X_en2.append(en2)
        X_pr2.append(pr2)
        # print(en1)
    return np.array(X_en1), np.array(X_pr1), np.array(X_en2), np.array(X_pr2)


X_en_tra, X_pr_tra, _, _ = get_data(enhancers_tra, promoters_tra)
X_en_imtra, X_pr_imtra, _, _ = get_data(im_enhancers_tra, im_promoters_tra)
X_en_tes, X_pr_tes, _, _ = get_data(enhancers_tes, promoters_tes)

"""
a npz file has 3 np array

read step:
0. data = np.load(npz_file)
1. data.files return an array key list
2. use data[key] or data[data.files[index]] to get np array

npz save:
np.savez(file_path,key1=np_array1,key2=np_array2,)
"""
np.savez(Data_dir + '%s_train.npz' % cell_name, X_en_tra=X_en_tra, X_pr_tra=X_pr_tra, y_tra=y_tra)
np.savez(Data_dir + 'im_%s_train.npz' % cell_name, X_en_tra=X_en_imtra, X_pr_tra=X_pr_imtra, y_tra=y_imtra)
np.savez(Data_dir + '%s_test.npz' % cell_name, X_en_tes=X_en_tes, X_pr_tes=X_pr_tes, y_tes=y_tes)
