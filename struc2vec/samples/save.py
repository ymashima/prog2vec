# coding:utf-8

"""推論結果を，csv形式(bz2で圧縮)で出力
"""

from logging import getLogger
from struc2vec.utils.log_setting import set_log

import os
import copy
import threading
import re
import textwrap

from struc2vec.ELS.read import ReadELS
from struc2vec.utils.file_control import Bz2file
from struc2vec.utils.file_control import Workspace
from struc2vec.utils.file_control import write_file
from struc2vec.utils.file_control import read_file
from struc2vec.utils.terminal import run_cmd
from struc2vec.model.feature_score import SVMScore
from struc2vec.model.infer import SVMPerf
from struc2vec.parse.java import java_source_code_pattern

LOGGER = getLogger(__name__)
set_log(LOGGER)


class SaveResult_SVM(object):
    """推論結果を，csv形式(bz2で圧縮)で出力

    Args:
        host (str): ホスト名
        port (str): ポート名
        index_name (str): インデックス名
    """

    def __init__(self, s2v_index, data_index, host, port, ftype):
        """

        Vars:
            self.ins_sample (ReadElasticsearch): ドキュメント名sampleの読み込みインスタンス
            self.ins_feature (ReadElasticsearch): ドキュメント名featureの読み込みインスタンス
            self.ins_symbol (ReadElasticsearch): ドキュメント名symbolの読み込みインスタンス
            self.ins_vector (ReadElasticsearch): ドキュメント名vectorの読み込みインスタンス
            self.class_id_dict (dict): class_idからclass_nameを参照するための辞書
            self.symbol_id_dict (dict): symbol_idからsymbol(記号)を参照するための辞書
            self.feature_id_dict (dict): feature_idからfeature(番号)を参照するための辞書
        """

        self.ins_sample = ReadELS(index=data_index, host=host, port=port, doc_type="sample")
        self.ins_feature = ReadELS(index=data_index, host=host, port=port, doc_type=ftype)
        self.ins_symbol = ReadELS(index=data_index, host=host, port=port, doc_type=ftype+"_symbol")
        self.ins_svm_model = ReadELS(index=s2v_index, host=host, port=port, doc_type="svm_model")
        self.ins_class_score = ReadELS(index=s2v_index, host=host, port=port, doc_type="class_score")
        self.class_id_dict = {}
        self.symbol_id_dict = {}
        self.feature_id_dict = {}
        self.feature_score_dict = {}
        self.search_feature_dict = {}
        self.not_search_feature_dict = {}
        # self.fs_list = [1,2,3,4,5]

        # method
        save_root = Workspace("result")
        self.__make_save_folder(os.path.join(save_root.get_path(), s2v_index))
        self.__make_class_id_dict()
        self.__make_symbol_id_dict()
        self.__make_feature_id_dict()
        # self.__make_feature_selection_list()

        # tmp
        self.gjc_path = "C:/Users/takelab/Documents/mashima/Data/code2xml_dot/finish/java"

    def __make_save_folder(self, name):
        """save folderの作成

        Args:
            name (str): 保存フォルダ名
        """

        save_folder = Workspace(name)
        self.save_root = save_folder.get_path()

    def __make_class_id_dict(self):
        """class_idからclass_nameを参照するための辞書の作成
        """

        for record in self.ins_sample.search_records(column = "class_id,class_name"):
            record = record["_source"]
            self.class_id_dict[record["class_id"]] = record["class_name"]
        # import pdb; pdb.set_trace()

    def __make_symbol_id_dict(self):
        """symbol_id_dictを作成
        """

        for symbol in self.ins_symbol.search_records(column="id,name,type"):
            symbol = symbol["_source"]
            self.symbol_id_dict[symbol["id"]] = "{}:{}".format(symbol["name"],symbol["type"])
        # import pdb; pdb.set_trace()

    def __make_feature_id_dict(self):
        """feature_id_dictの作成
        """

        for feature in self.ins_feature.search_records(column="id,name"):
            feature = feature["_source"]
            self.feature_id_dict[feature["id"]] = feature["name"]

    def __make_feature_selection_list(self):
        """self.fs_listの作成
        """

        self.fs_list = set([i["_source"]["k_fold"] for i in self.ins_svm_model.search_records(
            column="k_fold",
            body={'query': {'bool': {'filter': [
                {'term': {'class_id': 1}},
                {'term': {'feature_selection': 99999}},
            ]}}}
        )])

    def feature_file(self, feature_set):
        """feature_id_dictのcsvファイルを作成
        """

        ins_save = Bz2file(os.path.join(self.save_root, "feature.csv.bz2"))
        ins_save.add_data("feature_id,feature_name\n")
        for feature in sorted(feature_set):
            # feature_name = "/".join([self.symbol_id_dict[int(f)].split(":")[0] for f in self.feature_id_dict[feature].split("/")])
            feature_name = "/".join([self.symbol_id_dict[int(f)] for f in self.feature_id_dict[feature].split("/")])
            ins_save.add_data("{},{}\n".format(feature, feature_name,))
        ins_save.close_file()

    '''
    def evaluate_score(self):
        """評価値の保存

        Vars:
            ins_save (Bz2file): 保存圧縮ファイルのインスタンス
        """

        # init
        ins_save = Bz2file(os.path.join(self.save_root, "evaluate_score.csv.bz2"))
        ins_save.add_data("class_id,class_name,feature_selection,precision,recall,f1_score,accuracy\n")
        eval_dict = {class_id:{} for class_id in self.class_id_dict.keys()}
        for class_id in self.class_id_dict.keys():
            eval_dict[class_id] = {num_fs: {"pre": [], "rec": [], "f1": [], "acc": []}
                    for num_fs in self.fs_list}

        # write
        for class_score in self.ins_class_score.search_records(
                column="class_id,pre_svm,rec_svm,acc_svm,f1_svm,num_fs_svm"
        ):
            class_score = class_score["_source"]
            class_id = class_score["class_id"]
            write_data = "{cid},{cname},{fs},{pre},{rec},{f1},{acc}\n".format(
                cid=class_id,
                cname=self.class_id_dict[class_id],
                fs=class_score["num_fs_svm"],
                pre=class_score["pre_svm"],
                rec=class_score["rec_svm"],
                f1=class_score["f1_svm"],
                acc=class_score["acc_svm"],
            )
            ins_save.add_data(write_data)
        ins_save.close_file()
    '''

    def evaluate_score(self, num_fs):
        """評価値の保存

        Vars:
            ins_save (Bz2file): 保存圧縮ファイルのインスタンス
        """

        # init
        ins_save = Bz2file(os.path.join(self.save_root, "evaluate_score_fs{}.csv.bz2".format(num_fs)))
        ins_save.add_data("class_id,class_name,pre,rec,f1,acc\n")

        # write
        for score in self.ins_svm_model.search_records(
                column="class_id,pre,rec,acc,f1",
                body={'query': {'bool': {'filter': [
                        {'term': {'k_fold': "ave"}},
                        {'term': {'feature_selection': num_fs}},
                    ]}}},
                sort="class_id:asc",
        ):
            # import pdb; pdb.set_trace()
            score = score["_source"]
            class_id = score["class_id"]
            if class_id % 100 == 0:
                LOGGER.info("evaluate score class_id:%d", class_id)
            write_data = "{cid},{cname},{pre},{rec},{f1},{acc}\n".format(
                cid=class_id,
                cname=self.class_id_dict[class_id],
                pre=score["pre"],
                rec=score["rec"],
                f1=score["f1"],
                acc=score["acc"],
            )
            ins_save.add_data(write_data)
        ins_save.close_file()

    def evaluate_score_max(self):
        """評価値の保存

        Vars:
            ins_save (Bz2file): 保存圧縮ファイルのインスタンス
        """

        # init
        ins_save = Bz2file(os.path.join(self.save_root, "evaluate_score_max.csv.bz2"))
        ins_save.add_data("class_id,class_name,num_fs,pre,rec,f1,acc\n")

        # write
        for class_id in sorted(self.class_id_dict.keys()):
            records = self.ins_svm_model.search_records(
                    column="feature_selection,k_fold,pre,rec,acc,f1",
                    body={'query': {'bool': {'filter': [
                            {'term': {'class_id': class_id}},
                            {'term': {'k_fold': "ave"}},
                        ]}}},
                    sort="f1:desc,feature_selection:asc",
                    size=2, # fs:99999の場合は，2件目
            )
            try:
                score = records.__next__()["_source"]
                if score["feature_selection"] == 99999:
                    score = records.__next__()["_source"]
            except:
                import pdb; pdb.set_trace()
            if class_id % 100 == 0:
                LOGGER.info("evaluate score max class_id:%d", class_id)
            write_data = "{cid},{cname},{num_fs},{pre},{rec},{f1},{acc}\n".format(
                cid=class_id,
                cname=self.class_id_dict[class_id],
                num_fs=score["feature_selection"],
                pre=score["pre"],
                rec=score["rec"],
                f1=score["f1"],
                acc=score["acc"],
            )
            ins_save.add_data(write_data)
        ins_save.close_file()

    def dis_rep_class(self, work_path):
        """クラスごとに最も良い数値表現の保存(モデルのfeature_score)

        Vars:
            ins_save (Bz2file): 保存圧縮ファイルのインスタンス
            score_dict (dict): feature_idごとの推論結果を書き込む用のベース辞書
            class_score (dict): ドキュメント名vectorの検索結果
                class_id (int): クラスid
                k_fold (int): 交差検定の番号
                feature_selection (int): 使用した素性数
                feature_score (str): 素性ごとのscore
        """

        # init
        ins_save = Bz2file(os.path.join(self.save_root, "method_dis_rep.csv.bz2"))
        colum_list = ["class_id","class_name","svm_id","num_fs"]

        # 走査
        LOGGER.info("dis_rep_class make dict")
        self.feature_score_dict = {class_id:{} for class_id in self.class_id_dict.keys()}
        feature_set = set()
        for class_id in self.class_id_dict.keys():
            for score in self.ins_svm_model.search_records(
                column="class_id,k_fold,feature_selection,model_info",
                body={'query': {'bool': {'filter': [
                        {'term': {'class_id': class_id}},
                    ]}}},
                sort="f1:desc,feature_selection:asc",
            ):
                # if score["_source"]["feature_selection"] == 99999 or score["_source"]["k_fold"] == "ave":
                if score["_source"]["feature_selection"] in [100,1000,99999] or score["_source"]["k_fold"] == "ave":
                    continue
                self.feature_score_dict[class_id]["svm_id"] = score["_id"]
                self.feature_score_dict[class_id]["num_fs"] = score["_source"]["feature_selection"]
                # feature score
                write_file(path=os.path.join(work_path,"model.txt"), source=score["_source"]["model_info"])
                ins_svm_score = SVMScore(work_path=work_path)
                ins_svm_score.calc_score_from_model() # feature scoreの計算
                feature_score = str(ins_svm_score.get_score_str())
                self.feature_score_dict[class_id]["fscore"] = feature_score
                feature_set |= set([int(f.split(":")[0]) for f in feature_score.split(" ")])
                break

        # column list
        colum_list += [str(i) for i in sorted(feature_set)]
        ins_save.add_data(",".join(colum_list) + "\n")

        # feature.csv
        self.feature_file(feature_set) # feature_id_dictのcsvファイルの作成

        # write
        for class_id, fscore_dict in sorted(self.feature_score_dict.items()):
            tmp_score_dict = {f.split(":")[0]:f.split(":")[1] for f in fscore_dict["fscore"].split(" ")}
            write_data_list = [
                str(class_id),
                self.class_id_dict[class_id],
                str(fscore_dict["svm_id"]),
                str(fscore_dict["num_fs"]),
            ]
            if class_id % 100 == 0:
                LOGGER.info("dis_rep_class class_id:%d",class_id)
            for f in feature_set:
                f = str(f)
                if f in tmp_score_dict:
                    write_data_list.append(str(tmp_score_dict[f]))
                else:
                    write_data_list.append(str(0))
            ins_save.add_data(",".join(write_data_list) + "\n")
        # close
        ins_save.close_file()

    def plot_program_pattern(self):
        """program patternの可視化(主にroot_to_terminal)
        """

        # init
        save_folder = Workspace(directory=os.path.join(self.save_root, "program_pattern"))

        # make pattern
        for class_id in sorted(self.class_id_dict.keys()):
            # import pdb; pdb.set_trace()
            if class_id % 100 == 0:
                LOGGER.info("plot_program_pattern class_id:%d", class_id)
            feature_list = [
                f.split(":")[0]
                for f in self.feature_score_dict[class_id]["fscore"].split(" ")
                if float(f.split(":")[1]) >= 0
            ]
            not_feature_list = [
                f.split(":")[0]
                for f in self.feature_score_dict[class_id]["fscore"].split(" ")
                if float(f.split(":")[1]) < 0
            ]
            feature_set = set()
            symbol_set = set()
            self.search_feature_dict[class_id] = []  # source_code検索に使用
            self.not_search_feature_dict[class_id] = not_feature_list  # source_code検索に使用
            for f in feature_list:
                symbol_list = [s for s in self.feature_id_dict[int(f)].split("/")]
                self.search_feature_dict[class_id].append(tuple([len(symbol_list), f]))
                feature_set |= set([" -> ".join(symbol_list[i:(i + 2)]) for i in range(len(symbol_list) - 1)])
                symbol_set |= set([tuple([int(s), self.symbol_id_dict[int(s)].split(":")[0]]) for s in symbol_list])

            # plot
            output = self.__plot_dot(symbol_set=symbol_set, feature_set=feature_set)

            # write
            file_name = "{}_{}".format(class_id, self.class_id_dict[class_id])
            write_file(
                path=os.path.join(save_folder.get_path(), file_name+".dot"),
                source=output,
            )
            run_cmd("dot -Tpng {fname}.dot -o {fname}.png".format(fname=os.path.join(save_folder.get_path(), file_name)))
            run_cmd("dot -Teps {fname}.dot -o {fname}.eps".format(fname=os.path.join(save_folder.get_path(), file_name)))

    def __plot_dot(self, symbol_set, feature_set):
        """dotファイルの作成

        Args:
            symbol_set (set): symbol集合
            feature_set (set): feature集合

        Returns:
            (str): dotファイルの記述内容
        """

        output_str = textwrap.dedent('''
        Digraph {{
        //eps output: dot -Teps hogehoge.dot -o hogehoge.eps
        //png output: dot -Tpng hogehoge.dot -o hogehoge.png

        graph [
        // graph setting
        // rankdir = LR, // landscape on
        dpi = 150,
        charset = "UTF-8",
        fontcolor = white,
        layout = dot
        ];

        node [
        shape = box,
        //fontsize = 18
        ];

        // node define
        {node_list}

        // edge define
        {node}

        }}
        ''').format(
            node_list="".join(["{} [label=\"{}\"]\n".format(s, l)
                for s,l in sorted(symbol_set, key=lambda x:x[0])])+"\n",
            node="\n".join(feature_set)+"\n"
        ).strip()
        return output_str

    def search_sorce_code(self, work_path):
        """program patternに対応するソースコードの取得

        Args:
            work_path (str): tmp_workspaceのパス
        """

        # init
        save_folder = Workspace(directory=os.path.join(self.save_root, "source_code"))

        # make pattern
        for class_id in sorted(self.class_id_dict.keys()):
            # import pdb; pdb.set_trace()
            if class_id % 100 == 0:
                LOGGER.info("search_sorce_code class_id:%d", class_id)
            tmp_feature_list = sorted(self.search_feature_dict[class_id], key=lambda x: x[0])
            feature_list = []
            # feature_list = [tmp_feature_list[-1][1]]
            for count, f1 in enumerate(tmp_feature_list, start=0):
                flag = True
                f1_name = "/".join([s for s in self.feature_id_dict[int(f1[1])].split("/")])
                for f2 in tmp_feature_list[(count + 1):]:
                    f2_name = "/".join([s for s in self.feature_id_dict[int(f2[1])].split("/")])
                    if len(f2_name.replace(f1_name, "").split("/")) != f2[0]:
                        flag = False
                        break
                if flag:
                    feature_list.append(f1[1])

            # 最もpositiveが多く，negativeが少ないサンプルを一つ取り出す
            # import pdb; pdb.set_trace()
            base_body={'query': {'bool': {
                        'filter': [
                            {'term': {'class_id': class_id}},
                        ]}}}
            body = copy.deepcopy(base_body)
            # filter
            for f in feature_list:
                body['query']['bool']['filter'].append({"term":{"root_to_terminal": str(f)}})
            # must not
            body['query']['bool']['must_not'] = []
            for f in self.not_search_feature_dict[class_id]:
                body['query']['bool']['must_not'].append({'term': {'root_to_terminal': f}})
            records = self.ins_sample.search_records(
                column="class_id,class_name,sample_id,java,lines_of_code",
                body=body,
                sort="lines_of_code:asc",
                size=1,
            )
            try:
                record = records.__next__()["_source"]
            except StopIteration:
                body = copy.deepcopy(base_body)
                # filter
                for f in feature_list:
                    body['query']['bool']['filter'].append({"term":{"root_to_terminal": str(f)}})
                records = self.ins_sample.search_records(
                column="class_id,class_name,sample_id,java,root_to_terminal",
                body=body,
                sort="lines_of_code:asc",
                size=1,
            )
                try:
                    record = records.__next__()["_source"]
                except Exception as e:
                    LOGGER.warn("search_sorce_code second error:%s", e)
                    continue
            except Exception as e:
                LOGGER.warn("search_sorce_code first error:%s", e)
                continue

            # gjc java path
            # import pdb; pdb.set_trace()
            if record["java"] == "":
                java_path = os.path.join(self.gjc_path, "{}/{:04}.java".format(record["class_name"].replace("xml","java"),record["sample_id"]))
                record["java"], _ = read_file(path=java_path)

            # write file
            feature_name_list = []
            for f in feature_list:
                tmp_feature_name_list = []
                for s in self.feature_id_dict[int(f)].split("/"):
                    tmp_feature_name_list.append(self.symbol_id_dict[int(s)].split(":")[0])
                feature_name_list.append("/".join(tmp_feature_name_list))
            write_file(path=os.path.join(work_path, "source.java"), source=record["java"])
            write_file(path=os.path.join(work_path, "feature.txt"), source=",".join(feature_name_list))
            # import pdb; pdb.set_trace()

            # java
            java_source_code_pattern(
                java_path=os.path.join(work_path, "source.java"),
                xml_path=os.path.join(save_folder.get_path(), "{}.xml".format(record["class_name"])),
                feature_path=os.path.join(work_path, "feature.txt"),
            )

    '''
    def dis_rep_sample(self):
        """サンプルデータの数値表現の保存
        構築済みSVMモデルにおいて，Trueと判断されるモデルの中で，最もf1scoreが高く，
        使用する素性数が少ないモデルを選択

        Vars:
            ins_work (Workspace): 一時保存ファイルの書き出し用フォルダ
            ins_save (Bz2file): 保存圧縮ファイルのインスタンス
            score_dict (dict): feature_idごとの推論結果を書き込む用のベース辞書
            model_dict (dict): model_infoが書かれたdict
                model_info (str): SVMPerfのモデル情報
                model_id (str): モデルid (cls{}-fs{}-k{})
                feature_id_set (set): モデルが使用したfeature_idの集合
            sample_data (dict): ドキュメント名sampleの検索結果
                class_id (int): クラスid
                class_name (str): クラス名
                sample_id (int): サンプルid
                learning_data (str): 出現したfeature_id一覧 (SVMPerf用学習元データ)
        """

        # init
        ins_work = Workspace("tmp_workspace")
        ins_save = Bz2file(os.path.join(self.save_root, "sample_dis_rep.csv.bz2"))
        colum_list = ["class_id","class_name","sample_id", "model_id"]
        colum_list += [str(i) for i in range(1, len(self.feature_id_dict.keys()) + 1, 1)]
        ins_save.add_data(",".join(colum_list) + "\n")
        # score_dict = {fid: "0" for fid in self.feature_id_dict.keys()}

        # model infoの取得
        # model_dict = {}
        # for class_score in self.ins_vector.search_top_hits(
        #         column="class_id,k_fold,model_info,feature_score",
        #         group="class_id",
        #         size=1,
        #         sort_dict = {
        #             "f1": {"order": "desc"},
        #             "feature_selection": {"order": "asc"},
        #         },
        # ):
        #     model_id = class_score["class_id"]["hits"]["hits"][0]["_id"]
        #     class_score = class_score["class_id"]["hits"]["hits"][0]["_source"]
        #     class_id = class_score["class_id"]
        #     feature_id_set = set(map(int, re.sub(":[-|0-9|.|e]+","",class_score["feature_score"]).split(" ")))
        #     model_dict[class_id] = {
        #         "model_info": class_score["model_info"],
        #         "model_id": model_id,
        #         "feature_id_set": feature_id_set
        #     }

        # # all sample data
        # for sample_data in self.ins_sample.search_records(
        #         column="class_id,class_name,sample_id,learning_data",
        #         sort="class_id:asc,sample_id:asc"
        # ):
        #     sample_data = sample_data["_source"]
        #     sample_str = self.__calc_dis_rep_sample(
        #         sample_data=sample_data,
        #         model_dict=model_dict,
        #         ins_work=ins_work,
        #         score_dict=score_dict,
        #     )
        #     ins_save.add_data(sample_str)

        # close
        ins_save.close_file()
        ins_work.delete()

    def __calc_dis_rep_sample(self, model_dict, sample_data, ins_work, score_dict):
        """dis_rep_sampleメソッドの推論用内部メソッド

        Args:
            model_dict (dict): model_infoが書かれたdict
                model_info (str): SVMPerfのモデル情報
                model_id (str): モデルid (cls{}-fs{}-k{})
                feature_id_set (set): モデルが使用したfeature_idの集合
            sample_data (dict): ドキュメント名sampleの検索結果
                class_id (int): クラスid
                class_name (str): クラス名
                sample_id (int): サンプルid
                learning_data (str): 出現したfeature_id一覧 (SVMPerf用学習元データ)
            ins_work (Workspace): 一時保存ファイルの書き出し用フォルダ
                get_path(): 作成したフォルダのパスの取得
            score_dict (dict): feature_idごとの推論結果を書き込む用のベース辞書

        Returns:
            (str): サンプルデータの数値表現
        """

        class_id = sample_data["class_id"]
        sample_str_list = [
            str(class_id),
            str(sample_data["class_name"]),
            str(sample_data["sample_id"]),
            str(model_dict[sample_data["class_id"]]["model_id"])
        ]
        write_score_dict = copy.deepcopy(score_dict)

        path_dict = {
            "test": "{}/test.txt".format(ins_work.get_path()),
            "model": "{}/model.txt".format(ins_work.get_path()),
            "predict": "{}/predict.txt".format(ins_work.get_path()),
            "classify": "{}/classify.log".format(ins_work.get_path()),
        }
        # make model file
        write_file(path_dict["model"], model_dict[class_id]["model_info"])

        # make learning_data
        learning_id_list = [d for d in sample_data["learning_data"].split(" ") if d]
        if not learning_id_list:
            sample_str_list += list(score_dict.values())
            sample_str = ",".join(sample_str_list) + "\n"
            return sample_str

        learning_id_set = set(map(int, learning_id_list)) & model_dict[class_id]["feature_id_set"]
        learning_id_list = sorted(learning_id_set)
        test_data = "\n".join(["1 {d}:1 #{d}".format(d=d) for d in learning_id_list])
        source_dict = {"test": test_data,}

        # svmの計算と評価
        ins_svm = SVMPerf(path_dict=path_dict, source_dict=source_dict)
        ins_svm.run_classify()

        # add score
        sample_score = read_file(path_dict["predict"])[0]
        for learning_id, score in zip(learning_id_list, sample_score.split("\n")):
            write_score_dict[int(learning_id)] = score

        # write data
        # import pdb; pdb.set_trace()
        sample_str_list += [item[1] for item in sorted(write_score_dict.items())]
        sample_str = ",".join(sample_str_list) + "\n"

        # import pdb; pdb.set_trace()
        return sample_str
    '''