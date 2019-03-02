# coding:utf-8

"""素性評価に関するモジュール
"""

import os
from logging import getLogger

from struc2vec.utils.log_setting import set_log
from struc2vec.model.infer import SVMPerf
from struc2vec.utils.file_control import write_file, read_file

LOGGER = getLogger(__name__)
set_log(LOGGER)

class SVMScore(object):
    """SVMScoreを計算、1回計算するごとにインスタンスを生成

    Args:
        work_path (str): Workspaceのルートパス
    """

    def __init__(self, work_path):
        """

        Vars:
            self.path_dict (dict):
                test: (SVMスコア計算用) score_test.txtのパス
                model: (学習済みモデル) model.txtのパス
                predict: (SVMスコア計算用) score_predict.txtのパス
                classify: (SVMスコア計算用) score_classify.logのパス
            self.score_str (str): elasticsearchへSVMスコアを登録する専用変数
        """

        self.path_dict = {
            "test": "{}/score_test.txt".format(work_path),
            "model": "{}/model.txt".format(work_path),
            "predict": "{}/score_predict.txt".format(work_path),
            "classify": "{}/score_classify.log".format(work_path),
        }
        self.score_str = ""

    def calc_score_from_model(self):
        """model.txtファイルからSVMScoreを計算し，文字列として保存

        Vars:
            self.score_str (str): SVMScoreの文字列
        """

        model_data  = read_file(self.path_dict["model"])[0].split(os.linesep)
        bias = float(model_data[10].split(" ")[0])
        weight_vectors = [list(map(float, d.split(":"))) for d in model_data[11].split(" ")[1:-1]]
        feature_score = tuple([tuple([int(i[0]), i[1]-bias]) for i in weight_vectors])
        feature_score = sorted(feature_score, key=lambda x: x[1], reverse=True)
        self.score_str = " ".join([":".join(list(map(str, i))) for i in feature_score])


    def get_score_str(self):
        """self.score_strの取得

        Returns:
            self.score_str (str): svm score
        """

        return self.score_str
