import sys, os

root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])

from sequence_process.CKSNAP import CKSNAP
import numpy as np
import os

# In[]:
names = ['pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
cell_name = names[2]

feature_name = "cksnap"

train_dir = '../../data/epivan/%s/train/' % cell_name
imbltrain = '../../data/epivan/%s/imbltrain/' % cell_name
test_dir = '../../data/epivan/%s/test/' % cell_name
feature_dir = '../../data/epivan/%s/features/%s/' % (cell_name, feature_name)

if os.path.exists(feature_dir):
    print("path exits!")
else:
    os.mkdir(feature_dir)
    print("path created!")

print('Experiment on [%s] dataset' % cell_name)

print('Loading seq data...')
enhancers_tra = open(train_dir + '%s_enhancer.fasta' % cell_name, 'r').read().splitlines()[1::2]
promoters_tra = open(train_dir + '%s_promoter.fasta' % cell_name, 'r').read().splitlines()[1::2]
y_tra = np.loadtxt(train_dir + '%s_label.txt' % cell_name)
print("enhancers_tra:", len(enhancers_tra))
print("promoters_tra:", len(promoters_tra))
im_enhancers_tra = open(imbltrain + '%s_enhancer.fasta' % cell_name, 'r').read().splitlines()[1::2]
im_promoters_tra = open(imbltrain + '%s_promoter.fasta' % cell_name, 'r').read().splitlines()[1::2]
y_imtra = np.loadtxt(imbltrain + '%s_label.txt' % cell_name)
print("im_enhancers_tra:", len(im_enhancers_tra))
print("im_promoters_tra:", len(im_promoters_tra))
enhancers_tes = open(test_dir + '%s_enhancer_test.fasta' % cell_name, 'r').read().splitlines()[1::2]
promoters_tes = open(test_dir + '%s_promoter_test.fasta' % cell_name, 'r').read().splitlines()[1::2]
y_tes = np.loadtxt(test_dir + '%s_label_test.txt' % cell_name)
print("enhancers_tes:", len(enhancers_tes))
print("promoters_tes:", len(promoters_tes))

print('平衡训练集')
print('pos_samples:' + str(int(sum(y_tra))))
print('neg_samples:' + str(len(y_tra) - int(sum(y_tra))))
print('不平衡训练集')
print('pos_samples:' + str(int(sum(y_imtra))))
print('neg_samples:' + str(len(y_imtra) - int(sum(y_imtra))))
print('测试集')
print('pos_samples:' + str(int(sum(y_tes))))
print('neg_samples:' + str(len(y_tes) - int(sum(y_tes))))
