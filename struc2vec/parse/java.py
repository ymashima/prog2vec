# coding:utf-8

"""java解析
"""

from struc2vec.utils.terminal import run_jar
from struc2vec.utils.terminal import run_clang_format

def java_to_xml(java_path, xml_path):
    """javaファイルからxmlファイルを作成

    Args:
        java_path (str): javaファイルのパス
        xml_path (str): xmlファイルのパス

    Returns:
        return_code (int): 正常終了は0,エラーは負の値が戻り値になる
    """

    jar = "struc2vec/utils/code2xml_identifier.jar"
    return_code = run_jar(jar, java_path, xml_path)
    return return_code


def java_formatter(java_path):
    """clang_formatの実行によるjavaファイルの正規化

    Args:
        java_path (str): javaファイルのパス

    Returns:
        return_code (int): 正常終了は0,エラーは負の値が戻り値になる
    """

    return_code = run_clang_format(java_path)
    return return_code

def java_source_code_pattern(java_path, xml_path, feature_path):
    """javaファイルから素性に対応するソースコードの部分列を抽出

    Args:
        java_path (str): javaファイルのパス
        xml_path (str): xmlファイルのパス
        feature_path (str): 抽出したい素性(コンマ区切りで1行)

    Returns:
        return_code (int): 正常終了は0,エラーは負の値が戻り値になる
    """

    jar = "struc2vec/utils/source_code_pattern.jar"
    return_code = run_jar(jar, java_path, xml_path, feature_path)
    return return_code
