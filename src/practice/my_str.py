params = {'a': 2, 'd': 1, 'b': 2}
params_msg = ""
for k, v in params.items():
    params_msg += "{}={}, ".format(k, v)
print(params_msg)
params_msg = params_msg[: -2] + '; '
print("params_msg:", params_msg)
