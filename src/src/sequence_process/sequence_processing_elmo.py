import numpy as np
import os


def get_data_seq(enhancers, promoters):
    X_enpr = []
    for en, pro in zip(enhancers, promoters):
        X_enpr.append(en + ' ' + pro)
    return np.array(X_enpr)


# In[]:
names = ['pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
cell_name = names[4]
feature_name = "elmo"

train_dir = '../../data/epivan/%s/train/' % cell_name
imbltrain = '../../data/epivan/%s/imbltrain/' % cell_name
test_dir = '../../data/epivan/%s/test/' % cell_name
Data_dir = '../../data/epivan/%s/%s/' % (cell_name, feature_name)
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
"""
get npz file of 'en+,+pr'
"""
X_enpr_tra = get_data_seq(enhancers_tra, promoters_tra)
X_enpr_imtra = get_data_seq(im_enhancers_tra, im_promoters_tra)
X_enpr_tes = get_data_seq(enhancers_tes, promoters_tes)

"""
dir exist
"""
if not os.path.exists(Data_dir):
    print("\nData_dir is not exist,creating...")
    os.makedirs(Data_dir)
    print(os.path.abspath(Data_dir))

"""
save
"""
np.savez(Data_dir + '%s_train.npz' % cell_name, X_enpr_tra=X_enpr_tra, y_tra=y_tra)
print("save over!")
np.savez(Data_dir + 'im_%s_train.npz' % cell_name, X_enpr_tra=X_enpr_imtra, y_tra=y_imtra)
print("save over!")
np.savez(Data_dir + '%s_test.npz' % cell_name, X_enpr_tes=X_enpr_tes, y_tes=y_tes)
print("save over!")
