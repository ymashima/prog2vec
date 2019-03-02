# coding:utf-8

"""modelの性能評価モジュール
"""

from logging import getLogger
from struc2vec.utils.log_setting import set_log

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from struc2vec.utils.file_control import read_file

LOGGER = getLogger(__name__)
set_log(LOGGER)

class BinaryScore(object):
    """二群判別の評価用クラス
    """

    def __init__(self):
        """

        Vars:
            self.score_dict (dict): 評価結果の格納辞書
        """

        self.score_dict = {"acc": .0, "rec": .0, "pre": .0, "f1": .0}

    def calc_score(self, y_true, y_pred):
        """scoreの計算

        Args:
            y_true (list): 教師ラベルのリスト
            y_pred (list): 予測ラベルのリスト
        """

        self.score_dict["acc"] = accuracy_score(y_true=y_true, y_pred=y_pred)
        self.score_dict["rec"] = recall_score(y_true=y_true, y_pred=y_pred)
        self.score_dict["pre"] = precision_score(y_true=y_true, y_pred=y_pred)
        self.score_dict["f1"] = f1_score(y_true=y_true, y_pred=y_pred)

    def get_score(self, measure):
        """評価指標のscoreの取得

        Args:
            measure (str): acc, rec, pre, f1 のどれか

        Returns:
            self.score_dict[measure] (float): accuracy, recall, precision, f1_scoreのどれかのスコア
        """

        return self.score_dict[measure]

    def get_all_score(self):
        """全ての評価結果の取得

        Returns:
            self.score_dict (dict): 評価結果の辞書
        """

        return self.score_dict


class EvaluateSVMPerf(BinaryScore):
    """SVMPerfの評価

    Args:
        path_dict (dict): テストラベルと教師ラベルの読み込みパス
            test: テストデータのパス
            predict: テストデータの判別結果のパス
    """

    def __init__(self, path_dict):
        self.path_dict = path_dict
        super().__init__()

    def evaluate(self):
        """評価を行い，dictに格納する
        """

        y_true = []  # 教師ラベルのリスト
        y_pred = []  # 予測ラベルのリスト
        # import pdb; pdb.set_trace()
        try:
            for line in read_file(self.path_dict["test"])[0].strip().split("\n"):
                y_true.append(int(line.strip().split(" ")[0]))
            for line in read_file(self.path_dict["predict"])[0].strip().split("\n"):
                if float(line.strip()) >= 0:
                    y_pred.append(1)
                else:
                    y_pred.append(-1)
            self.calc_score(y_true=y_true, y_pred=y_pred)
            return 0  # 正常に処理ができた．
        except Exception as e:
            LOGGER.warn("evaluate error:%s", e)
            return - 1  # 失敗
