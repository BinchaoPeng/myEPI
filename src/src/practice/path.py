# In[ ]:
import os

print(os.getcwd())

names = ['pbc_IMR90', 'GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
cell_name = names[4]
feature_name = "elmo"
Data_dir = '../../data/epivan/%s/%s/' % (cell_name, feature_name)
print(os.path.abspath(Data_dir))
train_dir = '../../data/epivan/%s/train/' % cell_name
print(os.path.abspath(train_dir))

# In[ ]:
