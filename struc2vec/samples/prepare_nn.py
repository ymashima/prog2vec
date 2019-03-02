# coding:utf-8

"""neural networkの共通テーブル
"""

from logging import getLogger
import random
from scipy import sparse
import numpy as np
from joblib import Parallel, delayed

from struc2vec.utils.log_setting import set_log

LOGGER = getLogger(__name__)
set_log(LOGGER)

class CommonCrossValidationNN(object):
    """交差検定用にデータを整形するクラス（多群判別のみ）

    Args:
        index (str): インデックス名
        host (str): ホスト名
        port (str): ポート番号
        fold (str): 交差検定の分割数
        target_class_id (int): 判別対象のクラスid
        ins_common_id (CommonId): クラスid，サンプルid，feature_idに関するインスタンス
            get_class_id() (list): クラスidのリストを取得
            get_sample_id() (dictまたはlist): クラスidごとのサンプルidのリストを取得
        ins_common_score (CommonScore): SVMスコアに関するインスタンス
            get_fsn_id_dict() (dict): 素性選択用のfeature_idの辞書
    """

    def __init__(self,
        k_fold, ftype, num_core, class_id_list, class_learning_data_dict,
        num_feature, batchsize,
    ):
        """

        Vars:
            self.ins_sample (ReadElasticsearch): ドキュメント名sampleのReadElasticsearchインスタンス
            self.fold (int): 交差検定の分割数
            self.under (bool): アンダーサンプリングの有無
            self.target_class_id (int): 判別対象のクラスid
            self.sample_id_dict (dict): サンプルidのリスト
            self.fsn_id_dict　(dict): 素性選択用のfeature_id集合の辞書
            self.max_feature_id (int): feature_idの最大id
            self.feature_set (set): 素性選択で使用する素性id集合
        """

        # argument
        self.k_fold = k_fold
        self.ftype = ftype
        self.num_core = num_core
        self.batchsize = batchsize
        # self.target_class_id = target_class_id
        self.class_id_list = class_id_list
        self.class_learning_data_dict = class_learning_data_dict
        # self.max_feature_id = max_feature_id
        self.num_feature = num_feature
        # self.class_learning_data_dict = class_learning_data_dict
        # self.fsn_id_dict = svm_score_dict

        # vars
        self.fold_data_list = [[] for i in range(self.k_fold)]
        self.fold_target_list = [[] for i in range(self.k_fold)]
        # self.one_hot = np.eye(self.num_feature, dtype = np.float16)
        self.one_hot = sparse.eye(self.num_feature, dtype=np.int8, format="csr")
        self.batch_data_np = np.zeros((self.batchsize, self.num_feature), dtype=np.float16)
        # import pdb; pdb.set_trace()
        

        # self.feature_set = set()
        # self.sample_dict = {}

        # method
        self.__add_learning_data()
        # self.__add_class_records()
        LOGGER.info("setting ok")

    def yield_minibatch(self, data_type):
        """取得したtrainとtestをminibatchごとに出力させる

        Args:
            data_type (str): trainまたはtest
        """

        # import pdb; pdb.set_trace()
        for data, target in self.__yield_train_test(data_type=data_type):
            loop_count = -(-1 * len(target) // self.batchsize)  # 切り上げ
            for cnt in range(loop_count):
                batch_data = data[cnt * self.batchsize:(cnt + 1) * self.batchsize]
                batch_target = target[cnt * self.batchsize:(cnt + 1) * self.batchsize]
                batch_data_np, batch_target_np = self.__change_minibatch(batch_data=batch_data, batch_target=batch_target)
                # LOGGER.info("output batch")
                yield batch_data_np, batch_target_np

    def __change_minibatch(self, batch_data, batch_target):
        """minibatchのデータをnumpyに変換

        Args:
            batch_data (list): trainまたはtestのデータ
            batch_target (list): 教師データ

        Returns:
            (ndarray): ndarray変換した学習データと教師データ
        """

        # init
        # LOGGER.info("change minibatch")
        # LOGGER.info("zeros")
        self.batch_data_np = np.zeros((len(batch_data), self.num_feature), dtype=np.float16)

        # batch data
        # LOGGER.info("tmp_batch_data")
        tmp_batch_data = [np.array(list(map(int, data.split(" "))), dtype=np.int8)-1 for data in batch_data]  # feature_idは1から，しかしone-hotは0からに注意
        # one_hot_vectors = [self.one_hot[data] for data in tmp_batch_data]
        # import pdb; pdb.set_trace()
        # LOGGER.info("tmp_batch_data_np")
        # tmp_batch_data_np = [np.sum(self.one_hot[data].toarray(), axis=0, keepdims=True) for data in tmp_batch_data]  # こいつがくそ遅い
        tmp_batch_data_np = [sparse.csr_matrix.sum(self.one_hot[data], axis=0, dtype=np.int8).A1 for data in tmp_batch_data]  # こいつがくそ遅い
        # LOGGER.info("parallel change minibatch data")
        Parallel(n_jobs=self.num_core, verbose=0, backend="threading")([
            delayed(self.__parallel_change_minibatch_data)(tmp_batch_data_np=tmp_batch_data_np, count=count)
            for count in range(len(tmp_batch_data_np))
        ])

        # batch target
        # LOGGER.info("batch_target_np")
        batch_target_np = (np.array(batch_target, dtype=np.int8)-1)

        # LOGGER.info("end change minibatch")
        return self.batch_data_np, batch_target_np

    def __parallel_change_minibatch_data(self, tmp_batch_data_np, count):
        """numpyへの変換を並列処理

        Args:
            tmp_batch_data_np (ndarray): batchサイズのデータ
            count (int): 対応するサンプルid
        """

        self.batch_data_np[count] = tmp_batch_data_np[count]

    def __yield_train_test(self, data_type):
        """ELSからtrainとtestの元データを取得

        Args:
            data_type (str): trainまたはtest
        """

        for k in range(self.k_fold):
            train_data = []
            train_target = []
            test_data = []
            test_target = []
            for j in range(self.k_fold):
                if k == j:
                    test_data += self.fold_data_list[j]
                    test_target += self.fold_target_list[j]
                else:
                    train_data += self.fold_data_list[j]
                    train_target += self.fold_target_list[j]
            if data_type == "train":
                yield train_data, train_target
            else:
                yield test_data, test_target

    def __add_learning_data(self):
        """learning_dataを分割して確保
        """

        # samples
        # import pdb; pdb.set_trace()
        for class_id in self.class_id_list:
            samples = self.class_learning_data_dict[class_id]
            delimiter = -(-1*len(samples) // self.k_fold)  # 切り上げ
            random.shuffle(samples)
            for k in range(self.k_fold):
                self.fold_data_list[k] += [
                    sample["_source"][self.ftype]
                    for sample in samples[k*delimiter:(k + 1)*delimiter]
                ]
                self.fold_target_list[k] += [class_id for i in range(len(samples[k * delimiter:(k + 1) * delimiter]))]
                # import pdb; pdb.set_trace()
