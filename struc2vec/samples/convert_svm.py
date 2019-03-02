# coding:utf-8

"""vectorを行うための並列処理内容
"""

import os
import threading
import copy
from logging import getLogger
from joblib import Parallel
from joblib import delayed

from struc2vec.utils.log_setting import set_log
from struc2vec.utils.file_control import Workspace
from struc2vec.ELS.write import RegisterELS
from struc2vec.ELS.read import ReadELS
from struc2vec.model.evaluate import EvaluateSVMPerf
from struc2vec.model.feature_score import SVMScore
from struc2vec.model.infer import SVMPerf
from struc2vec.utils.file_control import read_file
from struc2vec.utils.file_control import write_file
# from struc2vec.samples.prepare_infer import CommonScore
from struc2vec.samples.prepare_svm import CommonSVMScore, CommonEvaluateScore
from struc2vec.samples.prepare_svm import CommonCrossValidation
from struc2vec.samples.prepare_svm import CommonLearningData

LOGGER = getLogger(__name__)
set_log(LOGGER)

class CalcSVM(object):
    """SVM計算を行うにあたり，事前準備を共有するためのクラス

    Args:
        object ([type]): [description]
    """

    def __init__(self, args, ftype, num_core, work_root ,ins_common_id):
        # args
        self.args = args
        self.under = self.args.under
        self.k_fold = self.args.cv
        self.ftype = ftype
        self.num_core = num_core
        self.work_root = work_root

        # vars
        self.feature_selection = self.args.fs

        # elasticsearch instance
        self.els_svm_model_write = RegisterELS(index=self.args.s2v_index, doc_type="svm_model", host=self.args.host, port=self.args.port,)
        self.els_svm_model_read = ReadELS(index=self.args.s2v_index, doc_type="svm_model", host=self.args.host, port=self.args.port,)
        self.els_class_score = RegisterELS(index=self.args.s2v_index, doc_type="class_score", host=self.args.host, port=self.args.port,)

        # common id
        self.class_id_list = ins_common_id.get_class_id_list()
        self.sample_id_dict = ins_common_id.get_sample_id_dict()
        self.max_feature_id = ins_common_id.get_max_feature_id(ftype=self.ftype)
        # common learning data
        ins_common_learning = CommonLearningData(index=self.args.data_index,host=self.args.host,port=self.args.port,ftype=self.ftype,class_id_list=self.class_id_list,)
        self.class_learning_data_dict = ins_common_learning.get_class_learning_data_dict()
        # common svm score
        self.ins_common_svm_score = CommonSVMScore(feature_selection=self.feature_selection,class_id_list=self.class_id_list,ftype=self.ftype,els_svm_model_read=self.els_svm_model_read,)
        # common evaluate score
        self.ins_common_evaluate_score = CommonEvaluateScore(feature_selection=self.feature_selection, class_id_list=self.class_id_list,)

    def __make_loop_list(self):
        """elasticsearchのbulk insertのために，2000件ずつ登録する
        """

        # init vars
        limit = 1000
        loop_class_id_list = []
        check = -(-1 * len(self.class_id_list) * self.k_fold // limit) # 切り上げ

        # loop list
        if check > 1:
            threshold = -(-1*limit // self.k_fold) # 切り上げ
            for count in range(check):
                loop_class_id_list = self.class_id_list[count * threshold:(count + 1) * threshold]
                if loop_class_id_list:
                    yield loop_class_id_list
        else:
            yield self.class_id_list

    def run(self):
        """class_idごとにParallelで計算

        methods:
            self.__calc_all_feature(): all featureを使ったSVM推論
            self.__calc_selected_feature(): selected featureを使ったSVM推論
            self.__calc_average(): 評価結果の平均値
            self.__calc_score(): ドキュメント名class_score
        """

        # all feature
        LOGGER.info("CalcSVM > run > start all feature")
        for class_id_list in self.__make_loop_list():
            Parallel(n_jobs=self.num_core, verbose=1, backend="threading")([
                delayed(self.__calc_all_feature)(class_id = class_id,ftype=self.ftype,)
                for class_id in class_id_list
            ])
            LOGGER.info("CalcSVM > run > register all feature")
            self.els_svm_model_write.register(mes="register all feature")

        # add feature score dict
        LOGGER.info("CalcSVM > run > add_positive_negative_feature")
        self.ins_common_svm_score.add_positive_negative_feature()

        # selected feature
        LOGGER.info("CalcSVM > run > start selected feature")
        for class_id_list in self.__make_loop_list():
            Parallel(n_jobs=self.num_core, verbose=1, backend="threading")([
                delayed(self.__calc_selected_feature)(class_id = class_id,ftype=self.ftype,)
                for class_id in class_id_list
            ])
            self.els_svm_model_write.register(mes="register selected feature")

        # average score
        LOGGER.info("CalcSVM > run > start average score")
        Parallel(n_jobs=self.num_core, verbose=1, backend="threading")([
            delayed(self.__calc_average)(class_id = class_id,ftype=self.ftype,)
            for class_id in self.class_id_list
        ])
        self.els_svm_model_write.register(mes="register average score")

        # class_score
        LOGGER.info("CalcSVM > run > start class_score")
        Parallel(n_jobs=self.num_core, verbose=1, backend="threading")([
            delayed(self.__calc_class_score)(class_id = class_id,ftype=self.ftype,)
            for class_id in self.class_id_list
        ])
        self.els_class_score.register(mes="register class_score")

    def __calc_all_feature(self, class_id, ftype):
        """all featureを使用したSVMでの計算

        Args:
            class_id (int): クラスid
            ftype (str): feature type
        """

        # logging
        LOGGER.info("ftype:%s, class_id:%d", ftype, class_id)
        # LOGGER.info("ftype:%s, class_id:%d > init", ftype, class_id)

        # init vars
        target_class_id = class_id
        # class_learning_data_dict = self.__prepare_learning_data_dict(target_class_id)

        # init instance
        ins_common_cross = CommonCrossValidation(
            k_fold=self.k_fold,
            target_class_id=target_class_id,
            ftype=self.ftype,
            class_id_list=self.class_id_list,
            sample_id_dict=self.sample_id_dict,
            max_feature_id=self.max_feature_id,
            class_learning_data_dict=self.class_learning_data_dict,
            # class_learning_data_dict=class_learning_data_dict,
            under=self.under,
        )

        # learn & innfer
        # LOGGER.info("ftype:%s, class_id:%d > start", ftype, class_id)
        feature_selection = [99999]  # all feature
        for train_test in ins_common_cross.yield_train_test(feature_selection):
            calc_vector_cv(
                args=self.args,
                class_id=target_class_id,
                els_svm_model_write=self.els_svm_model_write,
                work_root_path=self.work_root.get_path(),
                train_test=train_test,
                ftype=self.ftype,
                # ins_common_svm_score=self.ins_common_svm_score,
                ins_common_evaluate_score=self.ins_common_evaluate_score,
            )
        # LOGGER.info("ftype:%s, class_id:%d > end", ftype, class_id)

    def __calc_selected_feature(self, class_id, ftype):
        """selected featureを使用したSVMでの計算

        Args:
            class_id (int): クラスid
            ftype (str): feature type
        """

        # logging
        LOGGER.info("ftype:%s, class_id:%d", ftype, class_id)
        # LOGGER.info("ftype:%s, class_id:%d > init", ftype, class_id)

        # init vars
        target_class_id = class_id
        # class_learning_data_dict = self.__prepare_learning_data_dict(target_class_id)

        # init instance
        ins_common_cross = CommonCrossValidation(
            k_fold=self.k_fold,
            target_class_id=target_class_id,
            ftype=self.ftype,
            class_id_list=self.class_id_list,
            sample_id_dict=self.sample_id_dict,
            max_feature_id=self.max_feature_id,
            class_learning_data_dict=self.class_learning_data_dict,
            # class_learning_data_dict=class_learning_data_dict,
            svm_score_dict=self.ins_common_svm_score.get_svm_score_dict(target_class_id),
            under=self.under,
        )

        # learn & innfer
        # LOGGER.info("ftype:%s, class_id:%d > start", ftype, class_id)
        feature_selection = self.feature_selection  # selected feature
        for train_test in ins_common_cross.yield_train_test(feature_selection):
            calc_vector_cv(
                args=self.args,
                class_id=target_class_id,
                els_svm_model_write=self.els_svm_model_write,
                work_root_path=self.work_root.get_path(),
                train_test=train_test,
                ftype=self.ftype,
                ins_common_evaluate_score=self.ins_common_evaluate_score,
            )
        # LOGGER.info("ftype:%s, class_id:%d > end", ftype, class_id)

    def __calc_class_score(self, class_id, ftype):
        """ドキュメント名class_scoreの登録

        Args:
            class_id (int): クラスid
            ftype (str): feature type
            work_root_path (str): workspaceのルートフォルダ
            els_class_score (RegistElasticsearch): ドキュメント名class_scoreの登録用インスタンス
            els_svm_model_read (ReadElasticsearch): ドキュメント名svm_modelの参照用インスタンス
        """

        # logging
        LOGGER.info("ftype:%s, class_id:%d", ftype, class_id)

        # init
        work_thread = Workspace(directory = os.path.join(
                self.work_root.get_path(), "cls{}-fp{}".format(class_id, ftype)))

        # f1が最も高いrecordを取得
        records = self.els_svm_model_read.search_records(
            column="acc,pre,rec,f1,k_fold,feature_selection,model_info",
            body={'query': {'bool': {'filter': [
                        {'term': {'class_id': class_id}},
                        {'term': {'ftype': ftype}},
                    ]}}},
            sort="f1:desc,feature_selection:asc",
            size=2, # 1件目がaverageの場合は，2件目を参照
        )
        try:
            record = records.__next__()["_source"]
            if not record["model_info"]:
                record = records.__next__()["_source"]
        except:
            import pdb; pdb.set_trace()

        # feature score
        write_file(path=os.path.join(work_thread.get_path(),"model.txt"), source=record["model_info"])
        feature_score = calc_feature_score(work_thread=work_thread)

        # update
        record_id = "cls{}-fp{}".format(class_id, ftype)
        num_fs = record["feature_selection"]
        k_fold = record["k_fold"]
        svm_id = "cls{}-fs{}-k{}-fp{}".format(class_id, num_fs, k_fold, ftype)
        self.els_class_score.update(
            record_id=record_id,
            dict_source={
                "svm_id": svm_id,
                "feature_score_svm": feature_score,
                "num_fs_svm": num_fs,
                "acc_svm": record["acc"],
                "pre_svm": record["pre"],
                "rec_svm": record["rec"],
                "f1_svm": record["f1"],
            }
        )

        # delete
        work_thread.delete()

    def __calc_average(self, class_id, ftype):
        """推論結果の平均値の計算

        Args:
            args (namespace): プログラム実行時の引数
            class_id (int): クラスid
            num_fs (int): 素性選択の数
            ftype (str): feature type
            els_svm_model (RegistElasticsearch): ドキュメント名svm_modelの登録インスタンス
            ins_common_score (CommonScore): スコアに関する共通テーブルのインスタンス
        """

        # logging
        LOGGER.info("ftype:%s, class_id:%d", ftype, class_id)

        # init
        k_fold = "ave"

        # 登録
        feature_selection = self.feature_selection+[99999]
        for num_fs in feature_selection:
            # logging
            LOGGER.info("class: %d, fs: %s, k: %s", class_id, str(num_fs), str(k_fold))
            self.els_svm_model_write.create(
                record_id="cls{}-fs{}-k{}-fp{}".format(class_id, num_fs, k_fold, ftype),
                dict_source={
                    "setting": "c:{}".format(self.args.c_param),
                    "class_id": int(class_id),
                    "k_fold": str(k_fold),
                    "ftype": ftype,
                    "feature_selection": num_fs,
                    "num_train_id": "",
                    "num_test_id": "",
                    "acc": float(self.ins_common_evaluate_score.get_average_eval(
                        measure="acc", num_fs=num_fs, class_id=class_id)),
                    "pre": float(self.ins_common_evaluate_score.get_average_eval(
                        measure="pre", num_fs=num_fs, class_id=class_id)),
                    "rec": float(self.ins_common_evaluate_score.get_average_eval(
                        measure="rec", num_fs=num_fs, class_id=class_id)),
                    "f1": float(self.ins_common_evaluate_score.get_average_eval(
                        measure="f1", num_fs=num_fs, class_id=class_id)),
                    "feature_score": "",
                    "model_info": "",
                    "learn_log": "",
                    "predict": "",
                    "classify_log": "",
                    "train_id_set": "",
                    "test_id_set": "",
                }
            )

    def __prepare_learning_data_dict(self, class_id):
        """learning_data_dictをelasticsearchの検索でなんとかする場合

        Args:
            class_id ([type]): [description]

        Returns:
            [type]: [description]
        """

        # init vars
        target_class_id = class_id
        class_learning_data_dict = {class_id:[] for class_id in self.class_id_list}
        # elasticsearch instance
        els_sample_read = ReadELS(index=self.args.data_index, doc_type="sample", host=self.args.host, port=self.args.port,)

        # under sampling
        if self.under:
            # target_class_id_samples
            class_learning_data_dict[class_id] = [
                sample for sample in els_sample_read.search_records(
                    column="class_id,{}".format(self.ftype),
                    body={'query': {'bool': {'filter': [{'term': {'class_id': target_class_id}}]}}},
                )
            ]
            # not target_class_id_samples
            not_class_id_list = copy.deepcopy(self.class_id_list)
            not_class_id_list.remove(target_class_id)
            for sample in els_sample_read.search_top_hits(
                column="class_id,{}".format(self.ftype),
                group="class_id",
                size=self.k_fold,
                sort_dict={"lines_of_code":{"order":"desc"}}
            ):
                # import pdb; pdb.set_trace()
                class_id = sample["key"]
                if class_id in not_class_id_list:
                    class_learning_data_dict[class_id] = sample["class_id"]["hits"]["hits"]
        # not under sampling
        else:
            for sample in els_sample_read.search_records(
                column="class_id,{}".format(self.ftype),
            ):
                class_learning_data_dict[sample["_source"]["class_id"]].append(sample)

        return class_learning_data_dict


def calc_class_score(class_id, ftype, work_root_path, els_class_score, els_svm_model_read):
    """ドキュメント名class_scoreの登録

    Args:
        class_id (int): クラスid
        ftype (str): feature type
        work_root_path (str): workspaceのルートフォルダ
        els_class_score (RegistElasticsearch): ドキュメント名class_scoreの登録用インスタンス
        els_svm_model_read (ReadElasticsearch): ドキュメント名svm_modelの参照用インスタンス
    """

    # init
    work_thread = Workspace(directory=os.path.join(work_root_path, "cls{}-fp{}".format(class_id, ftype)))

    # f1が最も高いrecordを取得
    records = els_svm_model_read.search_records(
        column="acc,pre,rec,f1,k_fold,feature_selection,model_info",
        body={'query': {'bool': {'filter': [
                    {'term': {'class_id': class_id}},
                    {'term': {'ftype': ftype}},
                ]}}},
        sort="f1:desc,feature_selection:asc",
        size=2, # 1件目がaverageの場合は，2件目を参照
    )
    try:
        record = records.__next__()["_source"]
        if not record["model_info"]:
            record = records.__next__()["_source"]
    except:
        import pdb; pdb.set_trace()

    # feature score
    write_file(path=os.path.join(work_thread.get_path(),"model.txt"), source=record["model_info"])
    feature_score = calc_feature_score(work_thread=work_thread)

    # update
    record_id = "cls{}-fp{}".format(class_id, ftype)
    num_fs = record["feature_selection"]
    k_fold = record["k_fold"]
    svm_id = "cls{}-fs{}-k{}-fp{}".format(class_id, num_fs, k_fold, ftype)
    els_class_score.update(
        record_id=record_id,
        dict_source={
            "svm_id": svm_id,
            "feature_score_svm": feature_score,
            "num_fs_svm": num_fs,
            "acc_svm": record["acc"],
            "pre_svm": record["pre"],
            "rec_svm": record["rec"],
            "f1_svm": record["f1"],
        }
    )


def calc_vector_average(args, class_id, num_fs, ftype, els_svm_model, ins_common_score):
    """推論結果の平均値の計算

    Args:
        args (namespace): プログラム実行時の引数
        class_id (int): クラスid
        num_fs (int): 素性選択の数
        ftype (str): feature type
        els_svm_model (RegistElasticsearch): ドキュメント名svm_modelの登録インスタンス
        ins_common_score (CommonScore): スコアに関する共通テーブルのインスタンス
    """

    # init
    k_fold = "ave"
    if num_fs == 99999:
        ins_common_score.calc_f1_fscore(class_id=class_id, ftype=ftype)

    # ロギング
    LOGGER.info("class: %d, fs: %s, k: %s", class_id, str(num_fs), str(k_fold))

    # 登録
    els_svm_model.create(
        record_id="cls{}-fs{}-k{}-fp{}".format(class_id, num_fs, k_fold, ftype),
        dict_source={
            "setting": "c:{}".format(args.c_param),
            "class_id": int(class_id),
            "k_fold": str(k_fold),
            "ftype": ftype,
            "feature_selection": num_fs,
            "num_train_id": "",
            "num_test_id": "",
            "acc": float(ins_common_score.get_average_eval(measure="acc", num_fs=num_fs)),
            "pre": float(ins_common_score.get_average_eval(measure="pre", num_fs=num_fs)),
            "rec": float(ins_common_score.get_average_eval(measure="rec", num_fs=num_fs)),
            "f1": float(ins_common_score.get_average_eval(measure="f1", num_fs=num_fs)),
            "feature_score": "",
            "model_info": "",
            "learn_log": "",
            "predict": "",
            "classify_log": "",
            "train_id_set": "",
            "test_id_set": "",
        }
    )


def calc_vector_cv(
        args,
        class_id,
        els_svm_model_write,
        work_root_path,
        train_test,
        ftype,
        # ins_common_id,
        # ins_common_svm_score,
        ins_common_evaluate_score,
):
    """交差検定における推論

    Args:
        args (namespace): プログラム実行時の引数
        class_id (int): クラスid
        els_svm_model (RegistElasticsearch): ドキュメント名svm_modelの登録インスタンス
        work_root_path (str): workspaceのルートパス
        train_test (dict): 交差検定時の推論に必要なデータ辞書
        ftype (str): feature type
        ins_common_id (CommonID): idに関する共通テーブルのインスタンス
        ins_common_score (CommonScore): スコアに関する共通テーブルのインスタンス
    """

    # init
    num_fs = train_test["num_fs"]
    k_fold = train_test["k_fold"]
    feature_score = ""
    work_thread = Workspace(directory=os.path.join(work_root_path, "cls{}-fs{}-k{}-fp{}".format(class_id, num_fs, k_fold, ftype)))
    path_dict = {
        "train": "{}/train.txt".format(work_thread.get_path()),
        "test": "{}/test.txt".format(work_thread.get_path()),
        "model": "{}/model.txt".format(work_thread.get_path()),
        "learn": "{}/learn.log".format(work_thread.get_path()),
        "predict": "{}/predict.txt".format(work_thread.get_path()),
        "classify": "{}/classify.log".format(work_thread.get_path()),
    }
    source_dict = {
        "train": train_test["train_data"],
        "test": train_test["test_data"],
        "c_param": args.c_param,
    }

    # ロギング
    LOGGER.info("class: %d, fs: %s, k: %s start, [%s]", class_id, str(num_fs), str(k_fold),
                 "/{}-{}".format(os.getpid(), threading.get_ident()))

    # モデルの学習と推論
    ins_svm = SVMPerf(path_dict=path_dict, source_dict=source_dict)
    return_code = ins_svm.run_learn()
    if return_code >= 0:  # 正しく学習できた場合
        ins_svm.run_classify()

        # モデルの評価
        ins_eval_svm = EvaluateSVMPerf(path_dict=path_dict)
        return_code = ins_eval_svm.evaluate()
        if return_code >= 0:  # 正しくevaluateできた場合
            # ins_common_score.add_eval(
            ins_common_evaluate_score.add_eval(
                score_dict=ins_eval_svm.get_all_score(),
                num_fs = num_fs,
                class_id=class_id,
            )  # 評価値の追加

            # feature_scoreの計算
            if num_fs == 99999:
                feature_score = calc_feature_score(work_thread=work_thread)

            # elasticsearch辞書へ追加
            els_svm_model_write.create(
                record_id="cls{}-fs{}-k{}-fp{}".format(class_id, num_fs, k_fold, ftype),
                dict_source={
                    "setting": "c:{}".format(args.c_param),
                    "class_id": int(class_id),
                    "k_fold": str(k_fold),
                    "ftype": ftype,
                    "feature_selection": num_fs,
                    "num_train_id": len(train_test["train_id_list"]),
                    "num_test_id": len(train_test["test_id_list"]),
                    "acc": float(ins_eval_svm.get_score("acc")),
                    "pre": float(ins_eval_svm.get_score("pre")),
                    "rec": float(ins_eval_svm.get_score("rec")),
                    "f1": float(ins_eval_svm.get_score("f1")),
                    "feature_score": feature_score,
                    "model_info": read_file(path_dict["model"])[0],
                    "learn_log": read_file(path_dict["learn"])[0],
                    "predict": read_file(path_dict["predict"])[0],
                    "classify_log": read_file(path_dict["classify"])[0],
                    "train_id_set": " ".join(train_test["train_id_list"]),
                    "test_id_set": " ".join(train_test["test_id_list"]),
                }
            )
        else:  # evaluate失敗
            els_svm_model_write.create(
                record_id="cls{}-fs{}-k{}-fp{}".format(class_id, num_fs, k_fold, ftype),
                dict_source={
                    "setting": "c:{}".format(args.c_param),
                    "class_id": int(class_id),
                    "k_fold": str(k_fold),
                    "ftype": ftype,
                    "feature_selection": num_fs,
                    "num_train_id": 0,
                    "num_test_id": 0,
                    "acc": 0.0,
                    "pre": 0.0,
                    "rec": 0.0,
                    "f1": 0.0,
                    "feature_score": "",
                    "model_info": "evaluate error",
                    "learn_log": "",
                    "predict": "",
                    "classify_log": "",
                    "train_id_set": "",
                    "test_id_set": "",
                }
            )
    else:  # 学習が終わらなかった場合
        els_svm_model_write.create(
            record_id="cls{}-fs{}-k{}-fp{}".format(class_id, num_fs, k_fold, ftype),
            dict_source={
                "setting": "c:{}".format(args.c_param),
                "class_id": int(class_id),
                "k_fold": str(k_fold),
                "ftype": ftype,
                "feature_selection": num_fs,
                "num_train_id": 0,
                "num_test_id": 0,
                "acc": 0.0,
                "pre": 0.0,
                "rec": 0.0,
                "f1": 0.0,
                "feature_score": "",
                "model_info": "timeout error",
                "learn_log": "",
                "predict": "",
                "classify_log": "",
                "train_id_set": "",
                "test_id_set": "",
            }
        )

    # delete work_thread
    work_thread.delete()


def calc_feature_score(work_thread):
    """feature scoreの計算

    Args:
        work_thread (Workspace): threadのtmp_workspaceフォルダのインスタンス

    Returns:
        (str): feature scoreの文字列
    """

    ins_svm_score = SVMScore(work_path=work_thread.get_path())
    ins_svm_score.calc_score_from_model() # feature scoreの計算
    feature_score = str(ins_svm_score.get_score_str())
    return feature_score

