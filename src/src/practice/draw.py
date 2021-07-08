import matplotlib.pyplot as plt

N_EPOCHS = 10
train_auc_list = [1, 2, 3, 4, 5, 6, 76, 8, 9, 10]
test_auc_list = [6, 76, 8, 4, 5, 9, 10, 11, 2, 3, ]

train_auc_list1 = [1, 2, 3, 48, 9, 10, 5, 6, 76, 9]
test_auc_list1 = [6, 76, 8, 11, 2, 3, 4, 5, 9, 10, ]
x = range(1, N_EPOCHS + 1)

plt.plot(x, train_auc_list, 'c-', label="test_auc")
plt.plot(x, test_auc_list, 'm:', label="test_auc")
plt.ylabel("auc1")
plt.xlabel("epoch")
plt.legend()
plt.savefig("epoch_auc1.png", dpi=300)
plt.show()

plt.plot(x, train_auc_list1, 'g-', label="test_auc1")
plt.plot(x, test_auc_list1, 'r:', label="test_auc1")
plt.ylabel("auc2")
plt.xlabel("epoch")
plt.legend()
plt.savefig("epoch_auc2.png", dpi=300)
plt.show()

plt.plot(x, train_auc_list1, 'g-', label="test_auc1")
plt.plot(x, test_auc_list1, 'r:', label="test_auc1")
plt.plot(x, train_auc_list, 'c-', label="test_auc")
plt.plot(x, test_auc_list, 'm:', label="test_auc")
plt.ylabel("auc3")
plt.xlabel("epoch")
plt.legend()
plt.savefig("epoch_auc3.png", dpi=300)
plt.show()
