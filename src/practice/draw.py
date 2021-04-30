import matplotlib.pyplot as plt

N_EPOCHS = 10
test_auc_list = [1, 2, 3, 4, 5, 6, 76, 8, 9, 10]
x = range(1, N_EPOCHS + 1)
plt.plot(x, test_auc_list, 'b-o', label="test_auc")
plt.ylabel("auc")
plt.xlabel("epoch")
plt.savefig("epoch_auc.png", dpi=300)
plt.show()
