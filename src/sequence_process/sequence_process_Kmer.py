import os
import sys

import numpy as np

from sequence_process.Kmer import Kmer

root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])
from sequence_process.sequence_process_def import get_cell_line_seq

# In[]:
names = ['pbc_IMR90', 'GM12878', 'HeLa-S3', "HMEC", 'HUVEC', 'IMR90', 'K562', 'NHEK']
cell_name = names[7]
feature_name = "kmer"
data_source = "sept"

feature_dir, \
enhancers_tra, promoters_tra, y_tra, \
im_enhancers_tra, im_promoters_tra, \
y_imtra, enhancers_tes, promoters_tes, y_tes = get_cell_line_seq(data_source, cell_name, feature_name)


# In[ ]:

def get_data(enhancers, promoters):
    kmer = Kmer(n_jobs=1)
    X_en = kmer.run_Kmer(enhancers)
    # print(X_en)
    X_pr = kmer.run_Kmer(promoters)
    # print(X_pr)
    return np.array(X_en), np.array(X_pr)


"""
get and save
"""
X_en_tra, X_pr_tra = get_data(enhancers_tra, promoters_tra)
np.savez(feature_dir + '%s_train.npz' % cell_name, X_en_tra=X_en_tra, X_pr_tra=X_pr_tra, y_tra=y_tra)
if data_source == "epivan":
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
en_fasta [n,3000] ---> en_kmer [n,336]
pr_fasta [n,2000] ---> pr_kmer [n,336]

output:
en_kmer,pr_kmer,y

save:
npz: three np.array include en_kmer,pr_kmer,y

"""
