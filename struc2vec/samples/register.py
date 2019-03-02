# coding:utf-8

"""サンプルファイルの登録用
"""
from logging import getLogger
from elasticsearch import Elasticsearch
import shutil
import os
import threading
from joblib import Parallel
from joblib import delayed

from struc2vec.utils.log_setting import set_log
from struc2vec.parse.java import java_to_xml, java_formatter
from struc2vec.utils.file_control import Workspace, read_file
# from struc2vec.parse.xml import PathLength, RootToTerminal, TerminalToRoot
from struc2vec.parse.xml import PathLength, RootToTerminal
from struc2vec.ELS.read import ReadELS
from struc2vec.ELS.write import RegisterELS

LOGGER = getLogger(__name__)
set_log(LOGGER)

def register_sample(sample, args, work_root_path, ins_sample):
    """parallel用sampleをELSに登録(idに変換する前)

    Args:
        sample (dict): sample辞書
            class_name: クラス名
            class_id: クラスid
            sample_id: サンプルid
            sample_path: サンプルデータのファイルパス
        args (namespace): プログラム実行時の引数
            path (int): pathの長さ
            ext (str): 入力されるファイルの拡張子
            ft (boolean): clang_formatを実行するかどうか
        work_root_path (str): workspaceを作成するルートフォルダ
        ftype (str): 登録するfeatureの種類
        ins_sample (RegistElasticsearch): ドキュメント名sampleを作成するインスタンス
    """

    # init
    work_thread = Workspace(directory=os.path.join(work_root_path, "cls{}-sam{}".format(sample["class_id"], sample["sample_id"])))

    # xml
    # import pdb; pdb.set_trace()
    java_source = ""
    lines_of_code = 0
    xml_source = ""
    if sample["sample_id"] % 100 == 0:
        LOGGER.info("class:%d, sample:%d", sample["class_id"], sample["sample_id"])
    if args.ext == "java":
        shutil.copy2(sample["sample_path"], work_thread.get_path()+"/tmp.java")
        if args.ft:
            java_formatter(work_thread.get_path()+"/tmp.java")
        java_source, lines_of_code = read_file(work_thread.get_path() + "/tmp.java")
        java_to_xml(sample["sample_path"], work_thread.get_path()+"/tmp.xml")
    else:
        shutil.copy2(sample["sample_path"], work_thread.get_path()+"/tmp.xml")
    xml_source, xml_lines_of_code = read_file(work_thread.get_path() + "/tmp.xml")
    if lines_of_code == 0:
        lines_of_code = xml_lines_of_code # 入力データがxmlの場合は，xmlの行数を格納

    # add sample document
    # import pdb; pdb.set_trace()
    record_id = "{}-{}".format(sample["class_id"], sample["sample_id"])
    ins_sample.create(
        record_id=record_id,
        dict_source={
            "class_id": sample["class_id"],
            "class_name": sample["class_name"],
            "sample_id": sample["sample_id"],
            "java": java_source,
            "xml": xml_source,
            "dot": "",
            "path_length": "",
            "root_to_terminal": "",
            "terminal_to_root": "",
            "lines_of_code": lines_of_code,
        }
    )

    # delete work_thread
    work_thread.delete()
    # import pdb; pdb.set_trace()


def register_feature(length_feature_path, sample, ftype, ins_tmp_workspace, identifier):
    """parallelでfeatureをidに変換して，elasticsearchに登録する

    Args:
        length_feature_path (int): path_length_feature用のパスの長さ
        sample (dict): xmlを含むサンプルデータ
            class_id: クラスid
            sample_id: サンプルid
            xml: xmlデータ
        ftype (str): feature_type
        ins_tmp_workspace (RegisterElasticsearch): ドキュメント名tmp_workspaceのインスタンス
    """

    if sample["sample_id"] % 100 == 0:
        LOGGER.info("class:%d, sample:%d", sample["class_id"], sample["sample_id"])
    ins_parse_sample = ParseSample(length_feature_path=length_feature_path, ftype=ftype, identifier=identifier)
    learning_data = ins_parse_sample.run_tmp_learning_data(xml=sample["xml"])
    record_id = "{}-{}".format(sample["class_id"], sample["sample_id"])
    ins_tmp_workspace.create(
        record_id=record_id,
        dict_source={
            "class_id": sample["class_id"],
            "sample_id": sample["sample_id"],
            "learning_data": learning_data,
        }
    )


class ParseSample(object):
    """elasticsearchに格納するlearning用データを作成

    Args:
        ftype (str): 登録するfeatureの種類
        length_feature_path (int): ASTを区切るパスの長さ
    """

    def __init__(self, ftype, identifier, length_feature_path=0):
        """

        Vars:
            self.feature_dict (dict): データセットに出現するfeature辞書
            self.symbol_dict (dict): データセットに出現するsymbol辞書
            self.next_id (int): self.feature_dictを更新する際の次のid
            self.length_feature_path (int): ASTを区切るパスの長さ
        """

        self.ftype = ftype
        self.feature_dict = {}
        # self.symbol_dict = {}
        self.next_id = 1
        self.update_learning_dict = {}
        self.length_feature_path = length_feature_path
        self.identifier = identifier

    def run_tmp_learning_data(self, xml):
        """learning用データの作成

        Args:
            xml (str): xmlファイルのソース

        Vars:
            learning_data (str): svm_perfの学習用データ

        Returns:
            learning_data (str): svm_perfの学習用データ
        """

        learning_data = ""  # 出力するleaning_data変数
        if self.ftype == "path_length":
            ins_parse = PathLength(xml, self.length_feature_path, identifier=self.identifier)
        elif self.ftype == "root_to_terminal":
            ins_parse = RootToTerminal(xml, identifier=self.identifier)
        # elif self.ftype == "terminal_to_root":
        #     ins_parse = TerminalToRoot(xml)
        ins_parse.run_parse()
        learning_data = ins_parse.get_feature_str()

        # import pdb; pdb.set_trace()
        return learning_data  ## els_tmp_workspaceでcreate


class Feature(object):
    """一時保存用のlearning用データを加工し，アップデート
    (feature_id, symbol_idに置き換える)

    Args:
        index (str): インデックス名
        host (str): ホスト名
        port (str): ポート名
        ftype (str): 登録するfeatureの種類
        num_core (int): 並列処理を行うかどうか
    """

    def __init__(self, index, host, port, ftype, num_core):
        """

        Vars:
            self.els_tmp (Elasticsearch): ドキュメント名tmp_workspaceの削除用Elasticsearchインスタンス
            self.ins_tmp_workspace (ReadElasticsearch): ドキュメント名tmp_workspaceのReadElasticsearchインスタンス
            self.ins_sample_write (RegistElasticsearch): ドキュメント名sampleのRegistElasticsearchインスタンス
            self.ins_sample_read (ReadElasticsearch): ドキュメント名sampleのReadElasticsearchインスタンス
            self.ins_feature (RegistElasticsearch): ドキュメント名featureのRegistElasticsearchインスタンス
            self.ins_symbol (Symbol): Symbol登録用インスタンス

            self.feature_dict (dict): データセットに出現するfeature辞書
            self.feature_id_set (set): 使用するfeature_id集合
            self.next_id (int): feature辞書を更新する際の次のid
            self.limit (int): elasticsearchに一度に登録するデータの数
            self.num_core (int): 並列で行うかどうか
            self.count_feature_dict (dict): featureの出現頻度を調節する辞書
            self.frequency (int): featureの最低出現回数
        """

        self.feature_symbol_dict = {}  # feature_symbol: count
        self.feature_dict = {}  # feature: feature_id
        self.symbol_dict = {}
        self.limit = 5000
        self.num_core = num_core
        self.count_feature_dict = {}
        self.frequency = 10

        self.els_tmp = Elasticsearch(host=host, port=port)
        self.ins_tmp_workspace = ReadELS(
            index="tmp_workspace", host=host, port=port, doc_type="tmp_workspace")
        self.ins_sample_write = RegisterELS(
            index=index, host=host, port=port, doc_type="sample")
        self.ins_sample_read = ReadELS(
            index=index, host=host, port=port, doc_type="sample")
        self.ins_feature = RegisterELS(
            index=index, host=host, port=port, doc_type=ftype)
        self.ins_symbol = RegisterELS(
            index=index, host=host, port=port, doc_type=ftype+"_symbol")

    def update_learning_data(self, ftype):
        """ドキュメント名tmp_workspaceに格納されているlearning_dataを読み出し，
        feature_idとsymbol_idに変換した後に，
        ドキュメント名sampleにupdateする
        """

        # self.feature_dictの作成
        LOGGER.info("update_learning_data > make feature_symbol_dict (first)")
        for tmp_sample in self.ins_tmp_workspace.search_records(column="learning_data"):
            tmp_sample = tmp_sample["_source"]
            for feature_symbol in tmp_sample["learning_data"].split():
                if feature_symbol not in self.feature_symbol_dict:
                    self.feature_symbol_dict[feature_symbol] = 1
                else:
                    self.feature_symbol_dict[feature_symbol] += 1

        #  self.feature_dictの上書き, feature_symbol: feature_id
        LOGGER.info("update_learning_data > make feature_symbol_dict (second)")
        feature_symbol_set = set([feature_symbol
            for feature_symbol, count in self.feature_symbol_dict.items()
            if count >= self.frequency
        ])
        self.feature_symbol_dict = {feature_symbol: feature_id
            for feature_id, feature_symbol in enumerate(feature_symbol_set, start=1)
        }

        #  self.symbol_dict
        LOGGER.info("update_learning_data > make symbol_dict")
        symbol_set = set([symbol
            for feature in self.feature_symbol_dict.keys()
            for symbol in feature.split("/")
        ])
        self.symbol_dict = {symbol: symbol_id
            for symbol_id, symbol in enumerate(symbol_set, start=1)
        }

        #  self.feature_dict
        LOGGER.info("update_learning_data > make feature_dict")
        for feature_symbol, feature_id in self.feature_symbol_dict.items():
            feature = "/".join(list(map(str, [self.symbol_dict[symbol] for symbol in feature_symbol.split("/")])))
            self.feature_dict[feature] = feature_id

        #  再走査
        LOGGER.info("update_learning_data > update learning_data")
        for count, tmp_sample in enumerate(self.ins_tmp_workspace.search_records(column="learning_data"), start=1):
            learning_data_list = []
            for feature_symbol in tmp_sample["_source"]["learning_data"].split():
                if feature_symbol in self.feature_symbol_dict:
                    learning_data_list.append(self.feature_symbol_dict[feature_symbol])
            learning_data = " ".join(list(map(str, sorted(learning_data_list))))

            # update
            # import pdb; pdb.set_trace()
            record_id = tmp_sample["_id"]
            self.ins_sample_write.update(
                record_id=record_id,
                dict_source={
                    "{}".format(ftype): learning_data,
                }
            )
            if count % self.limit == 0:
                self.ins_sample_write.register(mes="update learning_data")
        self.ins_sample_write.register(mes="update learning_data (last)")

        # delete tmp_workspace documents
        LOGGER.info("update_learning_data > delete_tmp_workspace")
        self.__delete_tmp_workspace()

        # register
        LOGGER.info("update_learning_data > register_symbol")
        self.__register_symbol()

        LOGGER.info("update_learning_data > register_feature")
        self.__register_feature()

    def __register_symbol(self):
        #  self.symbol_dictの登録
        symbol_list = list(self.symbol_dict.items())
        for count in range(-(-1*len(symbol_list) // self.limit)):  # 切り上げ
            tmp_symbol_list = symbol_list[count * self.limit:(count + 1) * self.limit]
            Parallel(n_jobs=self.num_core, verbose=1, backend="threading")([
                delayed(self.__parallel_func_register_symbol)(
                    symbol=symbol,
                ) for symbol in tmp_symbol_list])
            self.ins_symbol.register()
        self.ins_symbol.register()

    def __parallel_func_register_symbol(self, symbol):
        """parallel用に定義した関数

        Args:
            feature (tuple):
                feature (str): featureの名前
                feature_id (int): featureのid
        """

        symbol, symbol_id = symbol
        symbol, symbol_type = symbol.split(":")
        self.ins_symbol.create(
            record_id=symbol_id,
            dict_source={
                "id": symbol_id,
                "name": symbol,
                "type": symbol_type,
            }
        )

    def __register_feature(self):
        """featureの登録
        """

        feature_list = list(self.feature_dict.items())
        for count in range(-(-1*len(feature_list) // self.limit)):  # 切り上げ
            tmp_feature_list = feature_list[count * self.limit:(count + 1) * self.limit]
            Parallel(n_jobs=self.num_core, verbose=1, backend="threading")([
                delayed(self.__parallel_func_register_feature)(
                    feature = feature,
                ) for feature in tmp_feature_list])
            self.ins_feature.register()
        self.ins_feature.register()

    def __parallel_func_register_feature(self, feature):
        """parallel用に定義した関数

        Args:
            feature (tuple):
                feature (str): featureの名前
                feature_id (int): featureのid
        """

        feature, feature_id = feature
        self.ins_feature.create(
            record_id=feature_id,
            dict_source={
                "id": feature_id,
                "name": feature,
                # "length": feature.count("/") + 1,
            }
        )

    def __delete_tmp_workspace(self):
        """ドキュメント名tmp_workspaceの削除
        """

        self.els_tmp.indices.delete(index="tmp_workspace", params={"ignore": 404})
