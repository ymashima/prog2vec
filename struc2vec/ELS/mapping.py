# coding:utf-8

"""elasticsearchのmapping(テーブル定義)に関するモジュール
"""

from elasticsearch import Elasticsearch


class MappingELS(object):
    """mappingを作成するクラス

    Args:
        index (str): インデックス名
        host (str): ホスト名
        port (str): ポート名
    """

    def __init__(self, index, host, port):
        """

        Vars:
            self.index (str): インデックス名
            self.els (Elasticsearch): elasticsearchのインスタンス
            self.json (dict): テーブル定義を格納する辞書
        """

        self.index = index
        self.els = Elasticsearch(host=host, port=port)
        self.json = {"mappings": {}}

    def create(self, table):
        """self.jsonにテーブル定義を追記する

        Args:
            table (dict): テーブル定義を行う，カラムとタイプの辞書
        """

        for doc_type, column_list in table.items():
            self.json["mappings"][doc_type] = {"properties": {}}
            for cname, ctype in column_list:
                column = {}
                if ctype == "long":
                    column = {"type": "long"}
                elif ctype == "float":
                    column = {"type": "float"}
                else:
                    column = {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    }
                self.json["mappings"][doc_type]["properties"][cname] = column

    def register(self):
        """indexにmappingを登録する
        """

        self.els.indices.delete(index=self.index, params={"ignore": 404})
        self.els.indices.create(index=self.index, body=self.json)

    def data_mapping2(self):
        """data用のmappingの作成と登録
        """

        table = {
            "sample": (  # サンプルデータ
                ("class_id", "long"),
                ("class_name", "text"),
                ("sample_id", "long"),
                ("java", "text"),
                ("xml", "text"),
                ("dot", "text"),
                ("path_length", "text"),  # path_length_feature_id_set: 学習用データ
                ("root_to_terminal", "text"),  # root_to_terminal_feature_id_set: 学習用データ
                # ("terminal_to_root", "text"),  # terminal_to_root_feature_id_set: 学習用データ
                ("lines_of_code", "long"),  # javaあるいはxmlの行数
            ),
            # nnの入力ユニットのためにidは連番で揃えておく
            "path_length": (  # ASTからpath1-5の長さの系列
                ("id", "long"),
                ("name", "text"),
                # ("length", "long"),  # pathの長さ
            ),
            "root_to_terminal": (  # ASTの根から葉までの系列の根を含む部分集合
                ("id", "long"),
                ("name", "text"),
                # ("length", "long"),
            ),
            # "terminal_to_root": (  # ASTの葉から根までの系列の葉を含む部分集合
            #     ("id", "long"),
            #     ("name", "text"),
            #     ("length", "long"),
            # ),
            # rnnの入力ユニットのためにidは連番で揃えておく
            "path_length_symbol": (  # 終端記号と非終端記号単体
                ("id", "long"),
                ("name", "text"),
                ("type", "text"),
            ),
            "root_to_terminal_symbol": (  # 終端記号と非終端記号単体
                ("id", "long"),
                ("name", "text"),
                ("type", "text"),
            ),
            # "terminal_to_root_symbol": (  # 終端記号と非終端記号単体
            #     ("id", "long"),
            #     ("name", "text"),
            #     ("type", "text"),
            # ),
        }

        self.create(table=table)

    def tmp_workspace_mapping(self):
        """一時保存用のmapping
        """

        table = {
            "tmp_workspace": (
                ("class_id", "long"),
                # ("class_name", "text"),
                ("sample_id", "long"),
                ("learning_data", "text"),
            )
        }

        self.create(table=table)

    def s2v_mapping(self):
        """vector用のmapping
        """

        table = {
            "sample_vector": ( # sampleの中間層出力
                ("class_id", "long"),
                ("class_name", "text"),
                ("sample_id", "long"),
                # ("ftype", "text"),  # feaature_type
                ("vec_nn", "text"),  # 中間層出力
                ("vec_rnn", "text"),
                ("dim_nn", "long"),  # 次元数
                ("dim_rnn", "long"),
            ),
            "symbol_vector": ( # symbolの中間層出力
                ("symbol_id", "long"),
                ("symbol_name", "text"),
                ("symbol_type", "text"),
                # ("ftype", "text"),  # feaature_type
                ("vec_nn", "text"), # 中間層出力
                ("vec_rnn", "text"),
                ("dim_nn", "long"), # 次元数
                ("dim_rnn", "long"),
                ("feature_id_nn", "text"), # 対応するfeature
                ("feature_id_rnn", "text"),
            ),
            "class_score": ( # classのscore(feature集合, plot)
                ("class_id", "long"),
                # ("class_name", "text"),
                ("svm_id", "text"),  # モデルid, cls{}_fs{}_k{}
                ("nn_id", "text"),  # モデルid, k_{}
                ("rnn_id", "text"),  # モデルid, k_{}
                ("ftype", "text"),  # feaature_type
                ("feature_score_svm", "text"),  # feature score集合
                ("feature_score_nn", "text"),
                ("feature_score_rnn", "text"),
                ("num_fs_svm", "long"),  # feature数
                ("num_fs_nn", "long"),
                ("num_fs_rnn", "long"),
                ("acc_svm", "float"), # 評価値:accuracy
                ("acc_nn", "float"),
                ("acc_rnn", "float"),
                ("pre_svm", "float"), # 評価値:precision
                ("pre_nn", "float"),
                ("pre_rnn", "float"),
                ("rec_svm", "float"), # 評価値:recall
                ("rec_nn", "float"),
                ("rec_rnn", "float"),
                ("f1_svm", "float"),  # 評価値:f1score
                ("f1_nn", "float"),
                ("f1_rnn", "float"),
            ),
            "svm_model": ( # 二値分類で構築したSVMのモデル
                ("setting", "text"),  # c:20,...
                ("class_id", "long"),
                ("k_fold", "text"),  # ave,1,2,..., 0:ave
                ("ftype", "text"),  # feaature_type
                ("feature_selection", "long"),  # 1,2,3,...,99999:all
                ("num_train_id", "long"),
                ("num_test_id", "long"),
                ("acc", "float"),
                ("pre", "float"),
                ("rec", "float"),
                ("f1", "float"),
                ("feature_score", "text"),  # all限定, best結果はclass_score
                ("model_info", "text"),  # svm_perf
                ("learn_log", "text"),
                ("predict", "text"),
                ("classify_log", "text"),
                ("train_id_set", "text"),
                ("test_id_set", "text"),
            ),
            "nn_model": (  # 多値分類で構築したneural network model
                ("setting", "text"),  # epochs:20,...
                ("k_fold", "text"),  # ave,1,2,..., 0:ave
                # ("ftype", "text"),  # feaature_type
                ("num_train_id", "long"), # 確認用
                ("num_test_id", "long"), # 確認用
                ("acc", "float"),
                ("f1_macro", "float"),
                ("f1_micro", "float"),
                ("feature_score", "text"),
                ("model_info", "text"),  # モデルの保存
                ("learn_log", "text"),  # 学習過程
                ("test_data", "text"),  # テストデータ
                ("predict", "text"),  # テストデータの予測結果
                ("train_id_set", "text"),
                ("test_id_set", "text"),
            ),
            "rnn_model": ( # 多値分類で構築したreccurent neural network model
                ("setting", "text"),  # epochs:20,...
                ("k_fold", "text"),  # ave,1,2,..., 0:ave
                # ("ftype", "text"),  # feaature_type
                ("num_train_id", "long"), # 確認用
                ("num_test_id", "long"), # 確認用
                ("acc", "float"),
                ("f1_macro", "float"),
                ("f1_micro", "float"),
                ("feature_score", "text"),
                ("model_info", "text"),  # モデルの保存
                ("learn_log", "text"),  # 学習過程
                ("test_data", "text"),  # テストデータ
                ("predict", "text"),  # テストデータの予測結果
                ("train_id_set", "text"),
                ("test_id_set", "text"),
            ),
        }

        self.create(table=table)
