import os
import sys

root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])

import numpy as np
import os


def get_cell_line_seq(data_source, cell_name, feature_name):
    if data_source != "epivan" and data_source != "sept":
        raise ValueError("data_source must be 'epivan' or 'sept' !")

    train_dir = '../../data/%s/%s/train/' % (data_source, cell_name)
    imbltrain = None
    if data_source == "epivan":
        imbltrain = '../../data/%s/%s/imbltrain/' % (data_source, cell_name)
    test_dir = '../../data/%s/%s/test/' % (data_source, cell_name)
    feature_dir = '../../data/%s/%s/features/%s/' % (data_source, cell_name, feature_name)

    if os.path.exists(feature_dir):
        print(feature_dir + " exits!")
    else:
        os.mkdir(feature_dir)
        print(feature_dir + " created!")

    print('Experiment on %s dataset' % cell_name)

    print('Loading seq data...')
    enhancers_tra = open(train_dir + '%s_enhancer.fasta' % cell_name, 'r').read().splitlines()[1::2]
    promoters_tra = open(train_dir + '%s_promoter.fasta' % cell_name, 'r').read().splitlines()[1::2]
    y_tra = np.loadtxt(train_dir + '%s_label.txt' % cell_name)

    im_enhancers_tra = None
    im_promoters_tra = None
    y_imtra = None
    if data_source == "epivan":
        im_enhancers_tra = open(imbltrain + '%s_enhancer.fasta' % cell_name, 'r').read().splitlines()[1::2]
        im_promoters_tra = open(imbltrain + '%s_promoter.fasta' % cell_name, 'r').read().splitlines()[1::2]
        y_imtra = np.loadtxt(imbltrain + '%s_label.txt' % cell_name)

    enhancers_tes = open(test_dir + '%s_enhancer_test.fasta' % cell_name, 'r').read().splitlines()[1::2]
    promoters_tes = open(test_dir + '%s_promoter_test.fasta' % cell_name, 'r').read().splitlines()[1::2]
    y_tes = np.loadtxt(test_dir + '%s_label_test.txt' % cell_name)

    print('平衡训练集')
    print('pos_samples:' + str(int(sum(y_tra))))
    print('neg_samples:' + str(len(y_tra) - int(sum(y_tra))))
    if data_source == "epivan":
        print('不平衡训练集')
        print('pos_samples:' + str(int(sum(y_imtra))))
        print('neg_samples:' + str(len(y_imtra) - int(sum(y_imtra))))
    print('测试集')
    print('pos_samples:' + str(int(sum(y_tes))))
    print('neg_samples:' + str(len(y_tes) - int(sum(y_tes))))
    if data_source == "epivan":
        return feature_dir, enhancers_tra, promoters_tra, y_tra, im_enhancers_tra, im_promoters_tra, y_imtra, enhancers_tes, promoters_tes, y_tes
    return feature_dir, enhancers_tra, promoters_tra, y_tra, enhancers_tes, promoters_tes, y_tes


if __name__ == '__main__':
    names = ['pbc_IMR90', 'GM12878', 'HeLa-S3', "HMEC", "HUVEC", 'IMR90', 'K562', 'NHEK']
    cell_name = names[3]
    feature_name = "test"
    get_cell_line_seq("sept", cell_name, feature_name)
