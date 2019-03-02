# coding:utf-8

"""
ターミナル上で実行する場合のソースコード
"""

from logging import getLogger
from struc2vec.utils.log_setting import set_log

import subprocess

LOGGER = getLogger(__name__)
set_log(LOGGER)

def run_cmd(cmd):
    """ターミナルでコマンドを実行

    Args:
        cmd (str): 実行するコマンド

    Returns:
        return_code (int): 正常終了は0,エラーは負の値が戻り値になる
    """

    timeout = 300 # 300秒(5分)
    popen_obj = subprocess.Popen(cmd, shell = True)
    try:
        subprocess.Popen.wait(popen_obj, timeout=timeout)
        return_code = popen_obj.returncode # 正常なら0
    except subprocess.TimeoutExpired:
        LOGGER.warn("subprocess.TimeoutExpired error")
        popen_obj.kill()
        return_code = -1
    return return_code

def run_jar(jar, java_path, xml_path, feture_path=None):
    """java -jarコマンドをターミナル上で実行する

    Args:
        jar (str): 実行するjarファイルのパス
        java_path (str): 解析するjavaファイルのパス
        xml_path (str): 解析結果のxmlファイルのパス

    Returns:
        return_code (int): 正常終了は0,エラーは負の値が戻り値になる
    """

    cmd = "java -jar {} {} {}".format(jar, java_path, xml_path)
    if feture_path:
        cmd = "java -jar {} {} {} {}".format(jar, java_path, xml_path, feture_path)
    return_code = run_cmd(cmd)
    return return_code

def run_clang_format(java_path):
    """clang-formatでjavaファイルのフォーマットを統一
    googleスタイルで，ファイルを上書きする．

    Args:
        java_path (str): clang_formatをかけるjavaファイルのパス

    Returns:
        return_code (int): 正常終了は0,エラーは負の値が戻り値になる
    """

    cmd = "clang-format -style=Google -i {}".format(java_path)
    return_code = run_cmd(cmd)
    return return_code
