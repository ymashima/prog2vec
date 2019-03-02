# coding:utf-8

"""elasticsearchからsampleのデータを取得するモジュール
"""

# import re
import copy
import random
from logging import getLogger

from struc2vec.utils.log_setting import set_log
from struc2vec.ELS.read import ReadELS
from struc2vec.ELS.write import RegisterELS

LOGGER = getLogger(__name__)
set_log(LOGGER)

class CommonId(object):
    """クラスidとサンプルidの共通の参照テーブル

    Args:
        host (str): elasticsearchのホスト名
        port (str): elasticsearchのポート名
        index (str): インデックス名
    """

    def __init__(self, host, port, index, ftype_list):
        """

        Vars:
            self.ins_search (ReadElasticsearch): sampleドキュメントのELSインスタンス
            self.ins_feature (ReadElasticsearch): featureドキュメントのELSインスタンス
            self.class_id_list (list): クラスidのリスト
            self.sample_id_dict (dict): サンプルidの辞書
            self.max_feature_id (int): feature辞書の最大feature_id
        """

        self.ftype_list = ftype_list
        self.ins_search = ReadELS(host=host, port=port, index=index, doc_type="sample")
        self.ins_ftype_dict = {
            ftype: ReadELS(host = host, port = port, index = index, doc_type = ftype)
            for ftype in self.ftype_list
        }
        self.class_id_list = []
        self.sample_id_dict = {}
        self.max_feature_id_dict = {ftype: {"max_id": 0, "size": 0} for ftype in self.ftype_list}

        # method
        self.__add_class_id() # self.class_id_listに追記
        self.__add_sample_id() # self.sample_id_dictに追記
        self.__add_feature_id() # self.feature_id_list

    def __add_class_id(self):
        """クラスidのリストを追記
        """
        for item in self.ins_search.search_top_hits(column="", group="class_id", size=1):
            self.class_id_list.append(item["key"])
        # import pdb; pdb.set_trace()

    def __add_sample_id(self):
        """クラスidごとにサンプルidのリストを追記
        """

        self.sample_id_dict = {class_id: [] for class_id in self.class_id_list}
        for record in self.ins_search.search_records(column="class_id,sample_id", sort="sample_id"):
            class_id = record["_source"]["class_id"]
            sample_id = record["_source"]["sample_id"]
            self.sample_id_dict[class_id].append(sample_id)

    def __add_feature_id(self):
        """feature辞書の追記
        """

        # self.max_feature_id = self.ins_feature.search_count()
        for ftype in self.ftype_list:
            records = self.ins_ftype_dict[ftype].search_records(column="id", sort="id:desc", size=1)
            self.max_feature_id_dict[ftype]["max_id"] = records.__next__()["_source"]["id"]
            self.max_feature_id_dict[ftype]["size"] = self.ins_ftype_dict[ftype].search_count()
        # import pdb; pdb.set_trace()

    def get_class_id_list(self):
        """クラスidのリストを取得

        Returns:
            self.class_id_list (list): クラスidのリスト
        """

        return self.class_id_list

    def get_sample_id_dict(self, class_id=None):
        """クラスidごとのサンプルidのリストを取得

        Args:
            class_id (int, optional): クラスidの指定，デフォルトはFalse

        Returns:
            self.sample_id_dict (dict): class_idごとのリスト（昇順）
            または
            self.sample_id_dict[class_id] (list): class_idに内包されているsample_idのリスト（昇順）
        """

        if class_id:
            return self.sample_id_dict[class_id]
        return self.sample_id_dict

    def get_max_feature_id(self, ftype):
        """feature辞書の最大feature_idを取得

        Returns:
            (int): feature_idの最大値 (feature_idの数)
        """

        return self.max_feature_id_dict[ftype]


class CommonEvaluateScore(object):
    """average用の評価値の辞書

    Args:
        feature_selection (list): 素性選択のリスト
    """

    def __init__(self, feature_selection, class_id_list):
        self.eval_dict = {class_id:{} for class_id in class_id_list}
        for class_id in class_id_list:
            for num_fs in feature_selection + [99999]:
                self.eval_dict[class_id][num_fs] = {"acc": [], "rec": [], "pre": [], "f1": []}# acc,rec,pre,f1の辞書

    def add_eval(self, score_dict, num_fs, class_id):
        """評価値の辞書の追加

        Args:
            class_id (int): クラスid
            score_dict (dict): acc,rec,pre,f1の辞書
            num_fs (int): 対象となる素性選択数(all:99999も含む)
        """

        for measure in score_dict.keys():
            self.eval_dict[class_id][num_fs][measure].append(score_dict[measure])

    def get_average_eval(self, measure, num_fs, class_id):
        """評価値の平均値を返す

        Args:
            class_id (int): クラスid
            measure (str): acc,rec,pre,f1のどれか
            num_fs (int): 対象となる素性選択数(all:99999も含む)

        Returns:
            (float): 評価値の平均値
        """

        eval_list = self.eval_dict[class_id][num_fs][measure]
        return sum(eval_list) / len(eval_list)


class CommonSVMScore(object):
    """SVMScoreの共通テーブル

    Args:
        index (str): インデックス名
        host (str): ホスト名
        port (str): ポート名
        feature_selection (list): allを除くfeature selectionのリスト
        class_id_list (list): クラスidのリスト
        ftype (str): feature type
    """

    def __init__(self, feature_selection, class_id_list, ftype, els_svm_model_read):
        # args
        self.ftype = ftype
        self.class_id_list = class_id_list
        self.feature_selection = feature_selection  # all:99999 なし

        # vars
        self.svm_score_dict = {class_id:{"posi":[], "nega":[]} for class_id in self.class_id_list}

        # elasticsearch instance
        # self.els_svm_model_read = ReadElasticsearch(host=host, port=port, index=index, doc_type="svm_model")
        self.els_svm_model_read = els_svm_model_read

    def add_positive_negative_feature(self):
        """k_foldの中でf1が最も高かったSVMスコアを使用

        Vars:
            feature_score_list (list): f1が最も高かったSVMスコアの一時保存リスト
        """

        for class_id in self.class_id_list:
            # 最も高いf1スコアの素性を使用
            records = self.els_svm_model_read.search_records(
                column="feature_score",
                body={'query': {'bool': {'filter': [
                            {'term': {'class_id': class_id}},
                            {'term': {'feature_selection': 99999}},
                            {'term': {'ftype': self.ftype}},
                        ]}}},
                sort="f1:desc",
                size=1,
            )
            try:
                record = records.__next__()["_source"]
            except:
                import pdb; pdb.set_trace()

            # add feature_score_list
            feature_score_list = []
            for feature in record["feature_score"].split(" "):
                feature_id, score = feature.split(":")
                feature_score_list.append(tuple([int(feature_id), float(score)]))
            self.feature_score_list = sorted(feature_score_list, key=lambda x: x[1], reverse=True)

            # add self.svm_score_dict
            max_fsn = max(self.feature_selection)
            self.svm_score_dict[class_id]["posi"] = [item[0] for item in self.feature_score_list[:max_fsn] if item[1] >= 0]
            self.svm_score_dict[class_id]["nega"] = [item[0] for item in self.feature_score_list[-1 * max_fsn:] if item[1] < 0]

    def get_svm_score_dict(self, class_id):
        """svm_score_dictの値をclass_idごとに取得

        Args:
            class_id (int): クラスid

        Returns:
            (dict):
                posi (list): positive featureのリスト
                nega (list): negative featureのリスト
        """

        return self.svm_score_dict[class_id]


class CommonLearningData(object):
    """sampleにあるlearning_dataを全てメモリにのせる

    Args:
        index (str): インデックス名
        host (str): ホスト名
        port (str): ポート番号
        ftype (str): feature type
        class_id_list (list): クラスidのリスト
    """

    def __init__(self, index, host, port, ftype, class_id_list):
        self.ins_sample = ReadELS(index=index, host=host, port=port, doc_type="sample")
        self.ftype = ftype
        self.class_learning_data_dict = {class_id:[] for class_id in class_id_list}

        # method
        self.__read_learning_data()

    def __read_learning_data(self):
        for sample in self.ins_sample.search_records(
            column="class_id,{}".format(self.ftype),
            sort="lines_of_code:desc"
        ):
            # sample = sample["_source"]
            class_id = sample["_source"]["class_id"]
            # learning_data = sample[self.ftype]
            self.class_learning_data_dict[class_id].append(sample)

    def get_class_learning_data_dict(self):
        return self.class_learning_data_dict


class CommonCrossValidation(object):
    """交差検定用にデータを整形するクラス（二群判別のみ）

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
        under (bool, optional): under samplingを有効にするかどうか，デフォルトはFalse
    """

    def __init__(
        self,
        k_fold,
        ftype,
        target_class_id,
        class_id_list,
        sample_id_dict,
        max_feature_id,
        class_learning_data_dict,
        svm_score_dict=None,
        under=False,
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
        self.target_class_id = target_class_id
        self.class_id_list = class_id_list
        self.sample_id_dict = sample_id_dict
        self.max_feature_id = max_feature_id
        self.class_learning_data_dict = class_learning_data_dict
        self.fsn_id_dict = svm_score_dict
        self.under = under

        # vars
        self.feature_set = set()
        # self.class_records = []
        self.sample_dict = {}
        self.target_class_id_records = []
        self.not_target_class_id_records = []

        # method
        self.__add_class_records()

    def __add_feature_set(self, num_fs):
        """素性選択で使用する素性id集合の作成

        Args:
            num_fs (int): 素性選択の数
        """

        # import pdb; pdb.set_trace()
        if num_fs == 99999:
            self.feature_set = [i for i in range(1, self.max_feature_id["max_id"] + 1, 1)]
        else:
            self.feature_set = set(self.fsn_id_dict["posi"][:num_fs] + self.fsn_id_dict["nega"][-1*num_fs:])

    def __add_class_records(self):
        """参照用サンプルデータをself.class_recordsに格納
        """

        # vars
        class_records = []

        # init
        self.target_class_id_records = self.class_learning_data_dict[self.target_class_id]
        class_id_list = copy.deepcopy(self.class_id_list)
        class_id_list.remove(self.target_class_id)

        # under sampling
        if self.under:
            self.not_target_class_id_records += [
                self.class_learning_data_dict[class_id][:self.k_fold]
                for class_id in class_id_list
            ]
        # not under sampling
        else:
            self.not_target_class_id_records += [
                self.class_learning_data_dict[class_id]
                for class_id in class_id_list
            ]

        # edit class records
        class_records += self.target_class_id_records
        class_records += [record for records in self.not_target_class_id_records for record in records]
        self.sample_dict = {sample["_id"]:sample["_source"][self.ftype] for sample in class_records}

        # import pdb; pdb.set_trace()

    def yield_train_test(self, feature_selection):
        """self.train_testにデータを追記

        Args:
            feature_selection (list): 素性選択の数の種類 # ここでallとそれ以外をコントロールする

        Vars:
            self.record_dict (ReadElasticsearch): ドキュメント名sampleのインスタンス
                learning_dataの取得
            train_test (dict): SVMPerfに登録&訓練・検証用のデータ辞書
                num_fs (int): 素性選択数,
                k_fold (int): 分割番号,
                train_data (str): SVMperf用訓練データ
                test_data (str): SVMperf用テストデータ,
                train_id_list (list): 訓練に使用したサンプルidのリスト,
                test_id_list (list): テストに使用したサンプルidのリスト,
        """

        for num_fs in feature_selection:
            self.__add_feature_set(num_fs) # 使用するfeature_idの集合を確保
            for k_fold, (train, test) in enumerate(self.get_train_test_id(self.target_class_id), start=1):
                # import pdb; pdb.set_trace()
                # LOGGER.info("num_fs:%d, k_fold:%d, start", num_fs, k_fold)
                # edit sample data
                learning_dict = {
                    sample_id: self.__edit_learning_data(learning_data=sample_data, target_class_id=self.target_class_id, num_fs=num_fs)
                    for sample_id, sample_data in self.sample_dict.items()
                }

                # train data
                train_data_list = []
                train_data_list += [
                    "1 {} #{}\n".format(learning_dict[sample_id], sample_id)
                    for sample_id in train["true"] if learning_dict[sample_id]
                ]
                train_data_list += [
                    "-1 {} #{}\n".format(learning_dict[sample_id], sample_id)
                    for sample_id in train["false"] if learning_dict[sample_id]
                ]

                # test data
                test_data_list = []
                test_data_list += [
                    "1 {} #{}\n".format(learning_dict[sample_id], sample_id)
                    for sample_id in test["true"] if learning_dict[sample_id]
                ]
                test_data_list += [
                    "-1 {} #{}\n".format(learning_dict[sample_id], sample_id)
                    for sample_id in test["false"] if learning_dict[sample_id]
                ]

                # yield dict
                train_test = {
                    "num_fs": num_fs,
                    "k_fold": k_fold,
                    # "train_data": "".join(train_data_list.values()),
                    # "test_data": "".join(test_data_list.values()),
                    # "train_id_list": list(train_data_list.keys()),
                    # "test_id_list": list(test_data_list.keys()),
                    "train_data": "".join(train_data_list),
                    "test_data": "".join(test_data_list),
                    "train_id_list": [],
                    "test_id_list": [],
                }
                # LOGGER.info("num_fs:%d, k_fold:%d, end", num_fs, k_fold)
                # if num_fs == 1:
                #    import pdb; pdb.set_trace()

                yield train_test

    def __edit_learning_data(self, learning_data, target_class_id, num_fs):
        """素性選択時の使用するfeature_idの調整

        Args:
            learning_data (str): SVMPerfに学習させるためのlearning_data

        Returns:
            learning_data (str): 引数に同じ
        """

        tmp_learning_data = [d for d in learning_data.strip().split(" ") if d]
        if not tmp_learning_data:
            learning_data = tmp_learning_data
            return learning_data

        learning_set = set(list(map(int, tmp_learning_data)))
        if num_fs == 99999:
            learning_list = sorted(tuple(map(int, learning_set)))
        else:
            learning_list = sorted(tuple(map(int, learning_set & self.feature_set)))
            # import pdb; pdb.set_trace()
        learning_data = " ".join(["{}:1".format(f) for f in learning_list])

        return learning_data

    def get_train_test_id(self, target_class_id):
        """サンプルidを訓練とテスト，true,falseに分けるための辞書の作成

        Yield:
            train (dict): 訓練に使用するサンプルidの辞書
                true (set): trueに該当するサンプルidの集合
                false (set): falseに該当するサンプルidの集合
            test (dict): テストに使用するサンプルidの辞書
                true (set): trueに該当するサンプルidの集合
                false (set): falseに該当するサンプルidの集合
        """

        true_list = self.__add_true_id(target_class_id)
        false_list = self.__add_false_id(target_class_id)
        for k in range(self.k_fold):
            train = {"true":set(), "false":set()}
            test = {"true":set(), "false":set()}
            for j in range(self.k_fold):
                if k == j:
                    test["true"] = test["true"].union(set(true_list[j]))
                    test["false"] = test["false"].union(set(false_list[j]))
                else:
                    train["true"] = train["true"].union(set(true_list[j]))
                    train["false"] = train["false"].union(set(false_list[j]))
            # import pdb; pdb.set_trace()
            yield train, test

    def __add_true_id(self, target_class_id):
        """判別結果がtrueのサンプルid集合の作成

        Args:
            target_class_id (int): 判別対象のクラスid

        Returns:
            (list): trueのサンプルid集合

        Vars:
            true_list (list): trueのサンプルid集合
            true_delimiter (int): 交差検定で分割するための境界番号
        """

        true_list = []
        true_delimiter = 0

        # import pdb; pdb.set_trace()
        true_tmp = copy.deepcopy(self.sample_id_dict[target_class_id])
        true_tmp = ["{}-{}".format(target_class_id, sample) for sample in true_tmp]
        random.shuffle(true_tmp)
        true_delimiter = round(len(true_tmp) / int(self.k_fold))
        for k in range(self.k_fold):
            if k == self.k_fold - 1:
                true_list.append(true_tmp[k * true_delimiter:])
            else:
                true_list.append(true_tmp[k * true_delimiter:(k + 1) * true_delimiter])

        return true_list

    def __add_false_id(self, target_class_id):
        """判別結果がfalseのサンプルid集合の作成

        Args:
            target_class_id (int): 判別対象のクラスid

        Returns:
            (list): falseのサンプルid集合

        Vars:
            false_list (list): falseのサンプルid集合
            false_delimiter (int): 交差検定で分割するための境界番号
        """

        false_list = [[] for i in range(self.k_fold)]
        false_delimiter = 0
        false_tmp_list = []

        for records in self.not_target_class_id_records:
            false_tmp_list.append([sample["_id"] for sample in records])
        # import pdb; pdb.set_trace()
        for false_tmp in false_tmp_list:
            random.shuffle(false_tmp)
            false_delimiter = int(len(false_tmp) / int(self.k_fold))
            for k in range(self.k_fold):
                if k == self.k_fold - 1:
                    false_list[k] += false_tmp[k * false_delimiter:]
                else:
                    false_list[k] += false_tmp[k * false_delimiter:(k + 1) * false_delimiter]
        # import pdb; pdb.set_trace()

        return false_list

