# coding:utf-8

"""neural networkでvector処理
"""

import os
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed

from logging import getLogger
from struc2vec.utils.log_setting import set_log
from struc2vec.ELS.read import ReadELS
from struc2vec.ELS.write import RegisterELS
from struc2vec.model.neural_net import NNModel
# from struc2vec.samples.prepare_svm import CommonLearningData
from struc2vec.samples.prepare_nn import CommonCrossValidationNN
from struc2vec.samples.prepare_svm import CommonLearningData
from struc2vec.utils.file_control import Workspace

# chainer
from chainer import Function, gradient_check, report, training, utils, Variable, initializers
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer.training import extensions
from chainer.backends.cuda import to_cpu, to_gpu
import chainer.functions as F
import chainer.links as L
from chainer import global_config

# sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

LOGGER = getLogger(__name__)
set_log(LOGGER)

class CalcNN(object):
    def __init__(self, args, ftype, num_core, work_root ,ins_common_id):
        # args
        self.args = args
        self.under = self.args.under
        self.k_fold = self.args.cv
        self.ftype = ftype
        self.num_core = num_core
        self.work_root = work_root

        # elasticsearch instance
        # self.els_ftype_read = ReadELS(index=self.args.data_index, doc_type=self.ftype, host=self.args.host, port=self.args.port,)
        self.els_sample_read = ReadELS(index=self.args.data_index, doc_type="sample", host=self.args.host, port=self.args.port,)
        self.els_nn_model_write = RegisterELS(index=self.args.s2v_index, doc_type="nn_model", host=self.args.host, port=self.args.port,)
        # self.els_svm_model_read = ReadELS(index=self.args.s2v_index, doc_type="svm_model", host=self.args.host, port=self.args.port,)
        self.els_class_score = RegisterELS(index=self.args.s2v_index, doc_type="class_score", host=self.args.host, port=self.args.port,)

        # common id
        self.class_id_list = ins_common_id.get_class_id_list()
        # self.sample_id_dict = ins_common_id.get_sample_id_dict()
        # self.sample_records = [sample for sample in self.els_sample_read.search_records(
        #     column="class_id,{}".format(self.ftype), sort="class_id:asc,sample_id:asc")]
        # self.max_feature_id = ins_common_id.get_max_feature_id()
        # common learning data
        ins_common_learning = CommonLearningData(index=self.args.data_index,host=self.args.host,port=self.args.port,ftype=self.ftype,class_id_list=self.class_id_list,)
        self.class_learning_data_dict = ins_common_learning.get_class_learning_data_dict()

        # vars
        self.feature_selection = self.args.fs
        self.num_feature = ins_common_id.get_max_feature_id(ftype=self.ftype)["size"]

        # vars nn model
        self.batchsize = 128
        self.n_in = self.num_feature
        self.n_mid = 100
        self.n_out = len(self.class_id_list)
        self.max_epoch = 100
        self.save_root = Workspace(directory="result_nn")
        self.gpu_id = 0  # CPUは -1, GPUは 0

        # common cross val
        self.ins_common_cross_nn = CommonCrossValidationNN(
            k_fold=self.k_fold, ftype=self.ftype, num_core=self.num_core,
            batchsize=self.batchsize, class_id_list=self.class_id_list,
            class_learning_data_dict=self.class_learning_data_dict, num_feature=self.num_feature
        )
        # # common svm score
        # self.ins_common_svm_score = CommonSVMScore(feature_selection=self.feature_selection,class_id_list=self.class_id_list,ftype=self.ftype,els_svm_model_read=self.els_svm_model_read,)
        # # common evaluate score
        # self.ins_common_evaluate_score = CommonEvaluateScore(feature_selection=self.feature_selection, class_id_list=self.class_id_list,)


    def run(self):
        self.__calc_all_feature()
        # self.__calc_selected_feature()
        # self.__calc_average()
        # self.__calc_class_score()

    def __calc_all_feature(self):
        for k in range(self.k_fold):
            # calc nn
            LOGGER.info("ftype%s, k_fold:%d", self.ftype, k)
            ins_nn = NN(
                    batchsize=self.batchsize, max_epoch=self.max_epoch, save_root_path=self.save_root.get_path(),
                    n_in=self.n_in, n_mid=self.n_mid, n_out=self.n_out, gpu_id=self.gpu_id,
                    ins_common_cross_nn=self.ins_common_cross_nn
                )
            # register
            ins_nn.learn()
            ins_nn.register_sample_vector()
            ins_nn.register_symbol_vector()
            ins_nn.register_class_score()

    def __calc_selected_feature(self):
        pass

    def __calc_class_score(self, class_id, ftype):
        pass

    def __calc_average(self, class_id, ftype):
        pass

    def __preprosessing(self):
        pass

    def make_model(self):
        pass



class NN(object):
    """推論を行うNNクラス

    Args:
        object ([type]): [description]
        batchsize (int): バッチサイズ
        max_epoch (int): 最大エポック数
        save_root_path (str): モデルを保存するルートパス
        n_id (int): 入力層のユニット数
        n_mid (int): 隠れ層のユニット数
        n_out (int): 出力層のユニット数
    """

    def __init__(self, batchsize, max_epoch, save_root_path, n_in, n_mid, n_out, gpu_id, ins_common_cross_nn):

        # args
        self.batchsize = batchsize
        self.max_epoch = max_epoch
        self.save_root_path = save_root_path
        self.n_in = n_in
        self.n_mid = n_mid
        self.n_out = n_out
        self.ins_common_cross_nn = ins_common_cross_nn

        # vars
        self.gpu_id = gpu_id  # Set to -1 if you use CPU
        self.feature_size = 0
        self.learning_data = {}
        self.class_learning_data_dict = {}
        self.one_hot_vector = 0  # ndarray
        self.feature_unit_id = {}  # feature_id: unit_id
        self.loss_scale = 128  # loss scale

        # method
        self.__set_model()

    def __set_model(self):
        """モデルの設定
        """

        # gpu
        self.model = NNModel(n_id=self.n_in, n_mid=self.n_mid, n_out=self.n_out)
        if self.gpu_id >= 0:
            self.model.to_gpu(self.gpu_id)

        # optimier
        self.optimizer = optimizers.Adam()
        self.optimizer.use_fp32_update()
        self.optimizer.setup(self.model)

    def learn(self):
        # one epoch training
        LOGGER.info("convert_nn >>> training start")
        for i in range(self.max_epoch):
            # train
            train_acc = []
            train_loss = []
            for j, (train, target) in enumerate(self.ins_common_cross_nn.yield_minibatch(data_type="train"), start=1):
                # prediction_train = self.model(to_gpu(batch_train))
                if self.gpu_id >= 0:
                    train = to_gpu(train)
                    target = to_gpu(target)
                # LOGGER.info("training model")
                prediction = self.model(train)  # 予測
                # LOGGER.info("calc loss")
                loss = F.softmax_cross_entropy(prediction, target)  # 損失関数
                train_loss.append(to_cpu(loss.data))
                # LOGGER.info("clear grads")
                self.model.cleargrads()
                # LOGGER.info("backward")
                loss.backward(loss_scale=self.loss_scale)
                # LOGGER.info("optimaize")
                self.optimizer.update()   # 勾配更新
                # LOGGER.info("acc")
                acc = F.accuracy(prediction, target)  # 正解率
                train_acc.append(to_cpu(acc.data))
                LOGGER.info("minibatch:%d, acc:%f, loss:%f", j, train_acc[-1], train_loss[-1])
            # import pdb; pdb.set_trace()

            # test
            test_loss = []
            test_acc = []
            for test, target in self.ins_common_cross_nn.yield_minibatch(data_type="test"):
                if self.gpu_id >= 0:
                    test = to_gpu(test)
                    target = to_gpu(target)
                prediction = self.model(test)  # 予測
                loss = F.softmax_cross_entropy(prediction, target)  # 損失関数
                test_loss.append(to_cpu(loss.data))
                acc = F.accuracy(prediction, target)  # 正解率
                test_acc.append(to_cpu(acc.data))
            # import pdb; pdb.set_trace()

            # logging
            LOGGER.info("epoch:%d, loss:%f, acc:%f, val_loss:%f, val_acc:%f",
                i+1, np.mean(train_loss), np.mean(train_acc), np.mean(test_loss), np.mean(test_acc))

            # import pdb; pdb.set_trace()
            # if i % 10 == 0:
            #     self.save_model(path=os.path.join(self.save_root_path, "{}_nn_model.npz".format(datetime.now().strftime("%Y%m%d_%H%M%S"))))
        # self.save_model(path=os.path.join(self.save_root_path, "{}_nn_model.npz".format(datetime.now().strftime("%Y%m%d_%H%M%S"))))
        # print(self.get_hidden_layer(self.test[0][0].reshape(1,-1)))  # 中間層出力

    def register_sample_vector(self):
        pass

    def register_symbol_vector(self):
        pass

    def register_class_score(self):
        pass

    def get_hidden_layer(self, x):
        return self.model.get_hidden_layer(x)

    def save_model(self, path):
        serializers.save_npz(path, self.model)
