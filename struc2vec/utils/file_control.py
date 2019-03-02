# coding:utf-8

"""
ファイル操作に関するモジュール
"""

from logging import getLogger
import re
import os
import threading
import shutil
import codecs
import bz2
import time
from random import Random  # seedをスレッドごとにするためのRandomクラス

from struc2vec.utils.log_setting import set_log

LOGGER = getLogger(__name__)
set_log(LOGGER)

def find_all_files(directory, ext):
    """拡張子extのファイルのパスを再帰的に取得

    Args:
        directory (str): ディレクトリのパス
        ext (str): 拡張子

    Yield:
        拡張子が付いたファイルの絶対パス
    """

    for root, _, files in os.walk(directory):
        for fdata in files:
            if re.search('.' + ext, fdata) is not None:
                yield os.path.abspath(os.path.join(root, fdata))


def scan_directory(root_dir, ext="java"):
    """input_directoryの走査
    sampleファイルを見つけるごとに，sample_info(dict)を返す．

    Args:
        root_dir (str): input_directory
        ext (str, optional): デフォルトは，"java". 拡張子

    Yield:
        sample_info (dict):
            class_name: クラス名,
            class_id: クラスid,
            sample_id: サンプルid,
            sample_path: サンプルファイルのパス
    """

    root_dir = root_dir.replace(os.sep, "/")  # osごとにパスの区切りを変更
    for root, dirs, _ in os.walk(root_dir):
        root = root.replace(os.sep, "/").replace(root_dir, "")
        root_info = {
            "list": root.split("/"),
            "rank": len(root.split("/")),
            "path": root_dir + root
        }
        if root_info["rank"] == 1:
            class_info = {
                "id": {name: id for id, name in enumerate(dirs, start=1)},
                "num": len(dirs)
            }
        elif root_info["rank"] == 2:
            for sample_id, sample_path in enumerate(
                    find_all_files(root_info["path"], ext), start=1):
                sample_path = sample_path.replace(os.sep, "/")
                class_name = root_info["list"][-1]
                class_id = class_info["id"][class_name]
                sample_info = {
                    "class_name": class_name,
                    "class_id": class_id,
                    "sample_id": sample_id,
                    "sample_path": sample_path
                }
                yield sample_info


def read_file(path):
    """ファイルの読み込み

    Args:
        path (str): 読み込むファイルのパス

    Returns:
        source (str): 読み込んだファイルの中身
        lines_of_data (int): データの行数
    """

    source = ""
    with codecs.open(path, "r", "utf-8") as fread:
        data = fread.readlines()
        lines_of_data = len(data)
        source = "".join(data)
    return source, lines_of_data


def write_file(path, source):
    """ファイルの書き込み

    Args:
        path (str): 書き込むファイルのパス
        source (str): 書き込む内容
    """

    with codecs.open(path, "w", "utf-8") as fwrite:
        fwrite.write(source)


class Workspace(object):
    """workspaceの作成と削除

    Args:
        root_dir (str): rootパスにworkspaceを作成
    """

    # def __init__(self, directory=None, root=False):
    def __init__(self, directory):
        """workspaceを作成

        Args:
            directory (str): スレッドのworkspaceのパス
        """

        self.directory = directory.replace(os.sep, "/")
        self.__make()

        # if root:
        #     self.directory = directory.replace(os.sep, "/")
        # else:
        #     seed = int(os.getpid()) + int(threading.get_ident())
        #     ins_random = Random(x=seed)
        #     self.directory = os.path.join(directory, str(ins_random.randrange(1,99999999)))

    def __make(self):
        """フォルダの作成, shutil.rmtreeでエラーが出るときがある．ちょっと考えないといけない
        """

        # if os.path.exists(self.directory):
        #     shutil.rmtree(self.directory, onerror=onerror)
        # os.mkdir(self.directory)

        try:
            if os.path.exists(self.directory):
                # shutil.rmtree(self.directory, onerror=onerror)
                shutil.rmtree(self.directory, ignore_errors=True)  # shutilのエラーを無視する
            os.mkdir(self.directory)
        except FileExistsError as f:
            # shutilのエラーにより，フォルダ作成ができない
            LOGGER.warn("FileExistsError %s", f)
            time.sleep(10)
            # import pdb; pdb.set_trace()
        except PermissionError as p:
            LOGGER.warn("PermissionError %s", p)
            time.sleep(10)
        except OSError as o:
            LOGGER.warn("OSError %s", o)
            time.sleep(10)

    def get_path(self):
        """格納しているdirectoryを取得
        """

        return self.directory

    def delete(self):
        """workspaceを削除
        """

        shutil.rmtree(self.directory, onerror=onerror)

    def refresh(self):
        """フォルダの再作成
        """

        self.delete()
        self.__make()


def onerror(func, path, exc_info):
    """
    Error handler for ``shutil.rmtree``.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.

    Usage : ``shutil.rmtree(path, onerror=onerror)``
    """

    import stat
    try:
        time.sleep(2) # ちょっと待ってみる
        LOGGER.warning("rmtree error")
        if not os.access(path, os.W_OK):
            LOGGER.warn("os.access(path, os.W_OK) == False")
            # Is the error an access error ?
            os.chmod(path, stat.S_IWUSR)
            func(path)
    except OSError as o:
        LOGGER.warn("os.access(path, os.W_OK) == True")
        LOGGER.warn("rmtree OSError {}".format(o))
        raise


class Bz2file(object):
    """bz2fileを作成するためのクラス

    Args:
        filename (str): 圧縮するファイル名
    """

    def __init__(self, filename):
        """

        Vars:
            self.bz2_file (bz2): 書き込むためのbz2オブジェクト
        """

        self.bz2_file = bz2.open(filename, "wb")

    def add_data(self, data, encoding="utf-8"):
        """データの書き込み

        Args:
            data (str): 圧縮する内容
            encoding (str, optional): Defaults to "utf-8". encodingの入力
        """

        data = data.encode(encoding)
        self.bz2_file.write(data)

    def close_file(self):
        """ファイルの書き込みの終了
        """

        self.bz2_file.close()
