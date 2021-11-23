from ML.ml_def import get_data_np_dict
import numpy as np

eiip_data = get_data_np_dict("epivan", "HUVEC", "eiip", "a")

print(eiip_data["train_X"].shape)
print(eiip_data["train_y"].shape)
print(eiip_data["test_X"].shape)
print(eiip_data["test_y"].shape)

# kmer_data = get_data_np_dict("epivan", "GM12878", "kmer", "a")
# eiip_data = get_data_np_dict("epivan", "GM12878", "eiip", "a")
# print(kmer_data["train_X"].shape)
# print(eiip_data["train_X"].shape)
# print(np.array(kmer_data["train_X"]).mean(axis=0))
# print(np.array(kmer_data["train_X"]).std(axis=0))
# print(np.array(eiip_data["train_X"][0]).mean(axis=0))
# print(np.array(eiip_data["train_X"][0]).std(axis=0))
# print(kmer_data["train_X"][0] - eiip_data["train_X"][0])
