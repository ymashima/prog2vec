# coding:utf-8

"""サンプルのELS登録を行う準備
"""

from struc2vec.utils.file_control import scan_directory
from struc2vec.ELS.read import ReadELS


class CommonSampleLoop(object):
    """Parallel処理のための繰り返し用配列

    Args:
        root_dir (str): データセットの入力パス
        ext (str): ファイルの拡張子
    """

    def __init__(self, root_dir, ext, index, host, port):
        """

        Vars:
            self.root_dir (str): データセットの入力パス
            self.ext (str): ファイルの拡張子
            self.limit (int): elasticsearchに一度に登録するデータの数
        """

        self.ins_sample = ReadELS(index=index, doc_type="sample", host=host, port=port)
        self.root_dir = root_dir
        self.ext = ext
        self.limit = 1000

    def yield_sample(self):
        """サンプルデータの出力

        Vars:
            sample_list (list): self.limit単位のsampleデータ

        Yield:
            sample_list (list): self.limit単位のsampleデータ
        """

        sample_list = []
        for count, sample in enumerate(scan_directory(
            root_dir=self.root_dir,
            ext=self.ext,
        ), start=1):
            sample_list.append(sample)
            if count % self.limit == 0:
                yield sample_list
                sample_list = []
        # smaple_listの要素数が1000個未満の時
        if sample_list:
            yield sample_list

    def yield_xml(self):
        """elasticsearchからのxmlデータの出力
        """

        sample_list = []
        for count,record in enumerate(self.ins_sample.search_records(
            column="xml,class_id,sample_id",
            sort="class_id:asc,sample_id:asc"
        ), start=1):
            sample_list.append({
                "class_id": record["_source"]["class_id"],
                "sample_id": record["_source"]["sample_id"],
                "xml": record["_source"]["xml"],
            })
            if count % self.limit == 0:
                yield sample_list
                sample_list = []
        # smaple_listの要素数が1000個未満の時
        if sample_list:
            yield sample_list
