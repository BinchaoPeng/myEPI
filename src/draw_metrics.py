import matplotlib.pyplot as plt


def drawMetrics(loss_list, test_auc_list, test_aupr_list, cell_name, feature_name, model_name):
    # polt
    x = range(1, len(test_auc_list) + 1)
    plt.plot(x, test_auc_list, 'g-', label="test_auc")
    plt.plot(x, test_aupr_list, 'r:', label="test_aupr")
    plt.ylabel("auc")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig("../model/%s_%s_%s_test.png" % (cell_name, feature_name, model_name.split()[-1]), dpi=300)
    plt.close()
    # plt.show()

    plt.plot(x, loss_list, 'c-', label="train_loss")
    # plt.plot(x, train_aupr_list, 'm:', label="train_aupr")
    plt.ylabel("aupr")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig("../model/%s_%s_%s_train_loss.png" % (cell_name, feature_name, model_name.split()[-1]), dpi=300)
    plt.close()
    plt.show()

    # plt.plot(x, test_auc_list, 'g-', label="test_auc")
    # plt.plot(x, train_auc_list, 'r:', label="train_auc")
    # plt.plot(x, test_aupr_list, 'c-', label="test_aupr")
    # plt.plot(x, train_aupr_list, 'm:', label="train_aupr")
    # plt.ylabel("auc & aupr")
    # plt.xlabel("epoch")
    # plt.legend()
    # plt.savefig("../model/%s_%s_%s_epoch_auc&aupr.png" % (cell_name, feature_name, model_name.split()[-1]), dpi=300)
    # plt.close()
    # plt.show()
