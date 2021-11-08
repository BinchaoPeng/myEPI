from ML.ml_def import get_data_np_dict

eiip_data = get_data_np_dict("epivan", "HUVEC", "tpcp", "a")

print(eiip_data["train_X"].shape)
print(eiip_data["train_y"].shape)
print(eiip_data["test_X"].shape)
print(eiip_data["test_y"].shape)

