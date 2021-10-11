import os
import sys
import time
import warnings

start_time = time.time()
warnings.filterwarnings("ignore")
root_path = os.path.abspath(os.path.dirname(__file__)).split('src')
sys.path.extend([root_path[0] + 'src'])

from sklearn.svm import SVC
from sklearnex import patch_sklearn

patch_sklearn()

"""
SVC参数解释
（1）C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0；
（2）kernel：参数选择有RBF, Linear, Poly, Sigmoid, 默认的是"RBF";
（3）degree：if you choose 'Poly' in param 2, this is effective, degree决定了多项式的最高次幂；
（4）gamma：核函数的系数('Poly', 'RBF' and 'Sigmoid'), 默认是gamma = 1 / n_features;
（5）coef0：核函数中的独立项，'RBF' and 'Poly'有效；
（6）probablity: 可能性估计是否使用(true or false)；是否启用概率估计。 这必须在调用fit()之前启用，并且会fit()方法速度变慢
（7）shrinking：是否进行启发式；
（8）tol（default = 1e - 3）: svm停止训练的误差精度;
（9）cache_size: 制定训练所需要的内存（以MB为单位）；默认为200MB。
（10）class_weight: 给每个类别分别设置不同的惩罚参数C，如果没有给，则会给所有类别都给C=1，即前面参数指出的参数C.如果给定参数‘balanced’，则使用y的值自动调整与输入数据中的类频率成反比的权重。缺省的话自适应；
（11）verbose: 是否启用详细输出。 此设置利用libsvm中的每个进程运行时设置，如果启用，可能无法在多线程上下文中正常工作。一般情况都设为False，不用管它。
（12）max_iter: 最大迭代次数，default = 1， if max_iter = -1, no limited;
（13）decision_function_shape ： ‘ovo’ 一对一, ‘ovr’ 多对多  or None 无, default=None
（14）random_state ：用于概率估计的数据重排时的伪随机数生成器的种子。
 ps：7,8,9一般不考虑。
 主要调节的参数有：C、kernel、degree、gamma、coef0。
 ★fit()方法：用于训练SVM，具体参数已经在定义SVC对象的时候给出了，这时候只需要给出数据集X和X对应的标签y即可。
 ★predict()方法：基于以上的训练，对预测样本T进行类别预测，因此只需要接收一个测试集T，该函数返回一个数组表示个测试样本的类别。
"""

from ML.ml_def import get_data_np_dict, time_since, get_scoring_result
from ML.EPIconst import EPIconst
from itertools import product

scoring = sorted(['f1', 'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy'])

for svm_item in product(EPIconst.CellName.all, EPIconst.FeatureName.all, [EPIconst.MethodName.svm]):
    start_time = time.time()
    cell_name, feature_name, method_name = svm_item
    data_list_dict = get_data_np_dict(cell_name, feature_name, method_name)
    clf = SVC(probability=True)  # 调参
    # print(clf.get_params().keys())
    clf.set_params(**EPIconst.ModelBaseParams.svm)
    cell_name_temp = cell_name + "_" + feature_name + "_" + method_name
    if cell_name.__contains__("HeLa-S3"):
        cell_name_temp = "HeLa_S3" + "_" + feature_name + "_" + method_name
    # print(cell_name_temp)
    model_params = getattr(EPIconst.ModelParams, cell_name_temp)

    clf.set_params(**model_params)
    clf.fit(data_list_dict['train_X'], data_list_dict['train_y'])
    y_pred = clf.predict(data_list_dict['test_X'])
    y_pred_prob_temp = clf.predict_proba(data_list_dict['test_X'])
    if (y_pred[0] == 1 and y_pred_prob_temp[0][0] >= 0.5) or (y_pred[0] == 0 and y_pred_prob_temp[0][0] < 0.5):
        y_proba = y_pred_prob_temp[:, 0]
    else:
        y_proba = y_pred_prob_temp[:, 1]

    process_msg, score_result_dict = get_scoring_result(scoring, data_list_dict["test_y"], y_pred, y_proba)
    print("{0}_{1}: {2} End time={3}\n".format(cell_name, feature_name, process_msg, time_since(start_time)))
