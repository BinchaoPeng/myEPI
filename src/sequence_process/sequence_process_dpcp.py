import sys, os
from sequence_process.physicalChemical import PhysicalChemical, PhysicalChemicalPath

from sequence_process.DPCP import DPCP
import numpy as np
import os

root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])

# In[]:
names = ['pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
cell_name = names[6]
feature_name = "dpcp"

train_dir = '../../data/epivan/%s/train/' % cell_name
imbltrain = '../../data/epivan/%s/imbltrain/' % cell_name
test_dir = '../../data/epivan/%s/test/' % cell_name
feature_dir = '../../data/epivan/%s/features/%s/' % (cell_name, feature_name)

if os.path.exists(feature_dir):
    print("path exits!")
else:
    os.mkdir(feature_dir)
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


pc_dict = PhysicalChemical(PhysicalChemicalPath.DiDNA_standardized).pc_dict

set_pc_list = ["Base stacking", "Protein induced deformability", "B-DNA twist", "A-philicity", "Propeller twist",
               "Duplex stability (freeenergy)", "Duplex stability (disruptenergy)", "DNA denaturation",
               "Bending stiffness", "Protein DNA twist", "Stabilising energy of Z-DNA", "Aida_BA_transition",
               "Breslauer_dG", "Breslauer_dH", "Breslauer_dS", "Electron_interaction", "Hartman_trans_free_energy",
               "Helix-Coil_transition", "Ivanov_BA_transition", "Lisser_BZ_transition", "Polar_interaction"]


#
# for idx, set_pc in enumerate(set_pc_list, 1):
#     print(idx, set_pc_list[idx - 1], pc_dict[set_pc])


def get_data(enhancers, promoters):
    dpcp = DPCP(2, set_pc_list, pc_dict, n_jobs=1)
    X_en = dpcp.run_DPCP(enhancers)
    # print(X_en)
    X_pr = dpcp.run_DPCP(promoters)
    # print(X_pr)
    return np.array(X_en), np.array(X_pr)


"""
get and save
"""
X_en_tra, X_pr_tra = get_data(enhancers_tra, promoters_tra)
np.savez(feature_dir + '%s_train.npz' % cell_name, X_en_tra=X_en_tra, X_pr_tra=X_pr_tra, y_tra=y_tra)
X_en_imtra, X_pr_imtra = get_data(im_enhancers_tra, im_promoters_tra)
np.savez(feature_dir + 'im_%s_train.npz' % cell_name, X_en_tra=X_en_imtra, X_pr_tra=X_pr_imtra, y_tra=y_imtra)
X_en_tes, X_pr_tes = get_data(enhancers_tes, promoters_tes)
np.savez(feature_dir + '%s_test.npz' % cell_name, X_en_tes=X_en_tes, X_pr_tes=X_pr_tes, y_tes=y_tes)

"""
a npz file has 3 np array

read step:
0. data = np.load(npz_file)
1. data.files return an array key list
2. use data[key] or data[data.files[index]] to get np array

npz save:
np.savez(file_path,key1=np_array1,key2=np_array2,)
"""
# np.savez(feature_dir + '%s_train.npz' % cell_name, X_en_tra=X_en_tra, X_pr_tra=X_pr_tra, y_tra=y_tra)
# np.savez(feature_dir + 'im_%s_train.npz' % cell_name, X_en_tra=X_en_imtra, X_pr_tra=X_pr_imtra, y_tra=y_imtra)
# np.savez(feature_dir + '%s_test.npz' % cell_name, X_en_tes=X_en_tes, X_pr_tes=X_pr_tes, y_tes=y_tes)

print("save over!")

"""
NOTE!!!

input: 
en_fasta
pr_fasta

method:
DPCP
set_pc_list

process:
en_fasta [n,3000] ---> en_cksnap [n,336]
pr_fasta [n,2000] ---> pr_cksnap [n,336]

output:
en_cksnap,pr_cksnap,y

save:
npz: three np.array include en_cksnap,pr_cksnap,y

"""
