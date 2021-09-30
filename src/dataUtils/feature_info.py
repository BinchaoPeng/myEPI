from ML.ml_def import get_data_np_dict

eiip_data = get_data_np_dict("NHEK", "eiip", "a")
kmer_data = get_data_np_dict("NHEK", "kmer", "a")
print(eiip_data["train_X"])
print(kmer_data["train_X"])
