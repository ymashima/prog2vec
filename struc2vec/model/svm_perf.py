# coding:utf-8
"""
svm perfに関するソースコード
"""
from struc2vec.utils.terminal import run_cmd

def learn(c_param, train_path, model_path, learn_path):
    """svmモデルを構築する関数

    Args:
        c_param (str): Cパラメータ
        train_path (str): 訓練データのパス
        model_path (str): 学習済みモデルのパス
        learn_path (str): 学習過程のログのパス

    Returns:
        return_code (int): 実行できたかどうかの確認, 0は成功, それ以外は失敗
    """

    cmd = "svm_perf_learn -c {} {} {} > {}".format(
        c_param, train_path, model_path, learn_path
    )
    return_code = run_cmd(cmd)
    return return_code

def classify(test_path, model_path, predict_path, classify_path):
    """学習済みsvmモデルで判別させる関数

    Args:
        test_path (str): テストデータのパス
        model_path (str): 学習済みモデルのパス
        predict_path (str): テストデータに対する判別結果のパス
        classify_path (str): 判別結果の総評のパス

    Returns:
        return_code (int): 実行できたかどうかの確認, 0は成功, それ以外は失敗
    """


    cmd = "svm_perf_classify {} {} {} > {}".format(
        test_path, model_path, predict_path, classify_path
    )
    return_code = run_cmd(cmd)
    return return_code
