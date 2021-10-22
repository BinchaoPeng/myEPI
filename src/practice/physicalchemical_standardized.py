import numpy as np
from sklearn.preprocessing import StandardScaler

AAA = [
    0.02 ,	0.02 ,	0.03 ,	0.02 ,	0.02 ,	0.02 	,0.02 ,	0.02 	,0.02 	,0.02 ,	0.02 	,0.02 ,	0.02 ,	0.02 ,	0.02 	,0.00


]
AAA = np.array(AAA)
std = np.std(AAA, axis=0)
mean = np.mean(AAA, axis=0)
#
# for item in AAA:
#     print((item - mean) / std)

scaler = StandardScaler()

a = AAA.reshape((AAA.size, -1))
# print(AAA)
scaler.fit(a)
print(scaler.mean_)
print(scaler.scale_)
print(scaler.var_)
print(scaler.transform(a))
