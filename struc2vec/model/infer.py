# coding:utf-8

"""inferモジュール
・推論を行うモジュール
"""

from struc2vec.model.svm_perf import learn, classify
from struc2vec.utils.file_control import write_file


class SVMPerf(object):
    """SVMperfを実行するクラス

    Args:
        path_dict (dict): pathに関する辞書
            train: 訓練データの入力パス
            test: テストデータの入力パス
            model: 学習済みモデルの出力パス
            learn: 学習過程のログの出力パス
            predict: テストデータの判別結果の出力パス
            classify: 判別結果の総評の出力パス
        source_dict (dict): dataに関する辞書
            train: 訓練データ
            test: テストデータ
            c_param: Cパラメータ
    """

    def __init__(self, path_dict, source_dict):
        self.path_dict = path_dict
        self.source_dict = source_dict

    def run_learn(self):
        """SVMperfの学習を行う

        Returns:
            return_code (int): 正常終了は0,エラーは負の値が戻り値になる
        """

        write_file(
            path=self.path_dict["train"],
            source=self.source_dict["train"]
        )
        return_code = learn(
            c_param=self.source_dict["c_param"],
            train_path=self.path_dict["train"],
            model_path=self.path_dict["model"],
            learn_path=self.path_dict["learn"]
        )

        return return_code

    def run_classify(self):
        """SVMperfの判別を行う

        Returns:
            return_code (int): 正常終了は0,エラーは負の値が戻り値になる
        """

        write_file(
            path=self.path_dict["test"],
            source=self.source_dict["test"]
        )
        return_code = classify(
            test_path=self.path_dict["test"],
            model_path=self.path_dict["model"],
            predict_path=self.path_dict["predict"],
            classify_path=self.path_dict["classify"]
        )

        return return_code
