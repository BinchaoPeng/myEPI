import os
import sys

root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])
from sequence_process.sequence_process_def import get_cell_line_seq
from sequence_process.PseKNC import PseTNC_II
import numpy as np

# In[]:
names = ['pbc_IMR90', 'GM12878', 'HeLa-S3', "HMEC", 'HUVEC', 'IMR90', 'K562', 'NHEK']
cell_name = names[7]
n_jobs = 1
lam = 5
W = 1
feature_name = "psetnc_II_lam%s_w%s" % (lam, W)
data_source = "sept"

feature_dir, \
enhancers_tra, promoters_tra, y_tra, \
im_enhancers_tra, im_promoters_tra, \
y_imtra, enhancers_tes, promoters_tes, y_tes = get_cell_line_seq(data_source, cell_name, feature_name)

# In[ ]:

set_pc_list = ["Bendability-DNAse",
               "Bendability-consensus",
               "Trinucleotide GC Content",
               "Nucleosome positioning",
               "Consensus_roll",
               "Dnase I",
               "Dnase I-Rigid",
               "MW-Daltons",
               "Nucleosome",
               "Nucleosome-Rigid",
               ]


def get_data(enhancers, promoters, n_jobs=n_jobs):
    psetnc = PseTNC_II(set_pc_list=set_pc_list, lam=lam, W=W, n_jobs=n_jobs)
    X_en = psetnc.run_PseTNC_II(enhancers)
    X_pr = psetnc.run_PseTNC_II(promoters)
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
psednc_II
set_pc_list

process:
en_fasta [n,3000] ---> en_cksnap [n,64+6*14]
pr_fasta [n,2000] ---> pr_cksnap [n,144]

output:
en_cksnap,pr_cksnap,y

save:
npz: three np.array include en_cksnap,pr_cksnap,y

"""
