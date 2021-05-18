import matplotlib.pyplot as plt


def drawMetrics(N_EPOCHS, train_auc_list, test_auc_list, train_aupr_list, test_aupr_list,
                cell_name, feature_name, model_name):
    # polt
    x = range(1, N_EPOCHS + 1)
    plt.plot(x, test_auc_list, 'g-', label="test_auc")
    plt.plot(x, train_auc_list, 'r:', label="train_auc")
    plt.ylabel("auc")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig("../model/%s_%s_%s_epoch_auc.png" % (cell_name, feature_name, model_name.split()[-1]), dpi=300)
    # plt.show()

    plt.plot(x, test_aupr_list, 'c-', label="test_aupr")
    plt.plot(x, train_aupr_list, 'm:', label="train_aupr")
    plt.ylabel("aupr")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig("../model/%s_%s_%s_epoch_aupr.png" % (cell_name, feature_name, model_name.split()[-1]), dpi=300)
    # plt.show()

    plt.plot(x, test_auc_list, 'g-', label="test_auc")
    plt.plot(x, train_auc_list, 'r:', label="train_auc")
    plt.plot(x, test_aupr_list, 'c-', label="test_aupr")
    plt.plot(x, train_aupr_list, 'm:', label="train_aupr")
    plt.ylabel("auc & aupr")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig("../model/%s_%s_%s_epoch_auc&aupr.png" % (cell_name, feature_name, model_name.split()[-1]), dpi=300)
    # plt.show()
