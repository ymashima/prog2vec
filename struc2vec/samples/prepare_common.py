# coding:utf-8

"""各モデルの共通の準備に関するモジュール
"""

from struc2vec.ELS.write import RegisterELS


class ClassScoreDoc(object):
    """ドキュメント名class_scoreに事前にデータを登録し，モデル結果をupdateする

    Args:
        index (str): インデックス名
        host (str): ホスト名
        port (str): ポート番号
    """

    def __init__(self, host, port, index):
        self.els_class_score = RegisterELS(
            host=host,
            port=port,
            index=index,
            doc_type="class_score"
        )

    def register_class_score(self, class_id_list, ftype_list):
        """class_scoreをelasticsearchに登録

        Args:
            class_id_list (list): クラスidのリスト
            ftype_list (list): feature typeのリスト
        """

        for ftype in ftype_list:
            for class_id in class_id_list:
                record_id = "cls{}-fp{}".format(class_id, ftype)
                self.els_class_score.create(
                    record_id=record_id,
                    dict_source={
                        "class_id": class_id,
                        # "class_name": class_name,
                        "svm_id": "",
                        "nn_id": "",
                        "rnn_id": "",
                        "ftype": ftype,
                        "fs_svm": "",
                        "fs_nn": "",
                        "fs_rnn": "",
                        "num_fs_svm": 0,
                        "num_fs_nn": 0,
                        "num_fs_rnn": 0,
                        "acc_svm": 0,
                        "acc_nn": 0,
                        "acc_rnn": 0,
                        "pre_svm": 0,
                        "pre_nn": 0,
                        "pre_rnn": 0,
                        "rec_svm": 0,
                        "rec_nn": 0,
                        "rec_rnn": 0,
                        "f1_svm": 0,
                        "f1_nn": 0,
                        "f1_rnn": 0,
                    }
                )
        self.els_class_score.register()
