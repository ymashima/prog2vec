# coding:utf-8

"""main関数
struc2vecが実行されたときに呼び出されるエントリーポイント

Example:
    python -m struc2vec -fs 10,20 -ft -cv 3 tests/test_data java test tests/save
"""


from datetime import datetime
from argparse import ArgumentParser
import textwrap
from logging import basicConfig, getLogger, config, DEBUG, INFO, StreamHandler, FileHandler, Formatter
from logging import getLogger, config
from joblib import Parallel
from joblib import delayed
import time

from struc2vec.utils.log_setting import set_log
from struc2vec.utils.file_control import Workspace
from struc2vec.ELS.mapping import MappingELS
from struc2vec.ELS.write import RegisterELS
from struc2vec.ELS.read import ReadELS
from struc2vec.samples.prepare_register import CommonSampleLoop
from struc2vec.samples.prepare_svm import CommonId
from struc2vec.samples.prepare_common import ClassScoreDoc
from struc2vec.samples.convert_svm import CalcSVM
from struc2vec.samples.convert_nn import CalcNN
from struc2vec.samples.register import register_sample, register_feature, Feature
from struc2vec.samples.save import SaveResult_SVM

# etc package
from struc2vec.etc.learning_data import frequency

# logging
LOGGER = getLogger(__name__)
set_log(LOGGER, root=True)

def main():
    """main関数
    struc2vecを実行するmain関数
    """

    struc2vec_description = textwrap.dedent("""
    struc2vecは，javaソースコード集合から，数値表現を獲得するプログラムです.
    ただし，feature_selectionの範囲:[1~90000]，all featureは，99999を表すことに注意してください
    """).strip()

    # positional arguments: 必須引数
    parser = ArgumentParser(description=struc2vec_description)
    parser.add_argument("mode", type=str, choices=["regi","vec","save","plot"], help="struc2vecのモード選択")
    parser.add_argument("data_index", type=str, help="データのインデックス名")
    parser.add_argument("s2v_index", type=str, help="作成する数値表現のインデックス名")
    # parser.add_argument("save_directory", type=str, help="数値表現の保存フォルダパス")

    # データの入力に関するオプション
    parser.add_argument("-dir", "--input_directory", default="", type=str,
                        help="ソースコードのフォルダーパス, デフォルトはなし",
                        metavar="PATH", dest="dir")
    parser.add_argument("-ext", "--extension", default="java", type=str, choices=["java", "xml"],
                        help="ディレクトリにあるファイルの種類, デフォルトはjava", dest="ext")
    parser.add_argument("-ft", "--clang_format", default=False, action="store_true",
                        help="clang_formatによるjavaソースコードの正規化, デフォルトはなし", dest="ft")

    # 数値表現の作成に関するオプション
    parser.add_argument("-fs", "--feature_selection", default=False, type=str,
                        help="選択する素性の数, e.g. -fs 1,10,100, デフォルトはなし",
                        metavar="LIST", dest="fs")
    parser.add_argument("-cv", "--cross_validation", default=5, type=int,
                        help="交差検定の分割数, e.g. -cv 5, デフォルトは5",
                        metavar="NUM", dest="cv")
    parser.add_argument("-u", "--under_sampling", default=False, action="store_true",
                        help="under samplingの有効化 デフォルトはFalse", dest="under")
    parser.add_argument("-m", "--model", default="svm", type=str, choices=["svm","nn"], # choices=["svm", "nn", "rnn"],
                        help="機械学習モデルの選択, デフォルトはsvm",dest="model")
    parser.add_argument("-c", "--c_param", default=20, type=int,
                        help="SVMのCパラメータ, e.g. -c 20, デフォルトは20", metavar="NUM")

    # ファイルの書き込み
    parser.add_argument("-ns", "--not_save", default=False, action="store_true",
                        help="推論結果をファイルへ書き込まない, デフォルトはなし")
    # 調査系統
    parser.add_argument("-sl", "--statistics_learning_data", default=False, action="store_true",
                        help="learning_dataの統計情報, デフォルトはなし")

    # 共通設定
    parser.add_argument("-p", "--parallel", default=False, action="store_true",
                        help="データ登録と推論の処理において，並列化を行う．デフォルトはFalse")
    parser.add_argument("-l", "--length_path", default=5, type=int,
                        help="ASTのpathの長さ e.g. -p 5, デフォルトは5", metavar="NUM", dest="path")
    parser.add_argument("-i", "--identifier", default=False, action="store_true",
                        help="終端記号(識別子)を含むかどうか．デフォルトはFalse")
    parser.add_argument("-ftype", default="root_to_terminal", type=str, choices=["root_to_terminal", "path_length"],
                        help="feature type, デフォルトはroot_to_terminal")

    # elasticsearchに関する設定
    parser.add_argument("-host", "--els_host", default="127.0.0.1", type=str,
                        help="elasticsearchのipアドレス, e.g. 127.0.0.1", dest="host")
    parser.add_argument("-port", "--els_port", default=9200, type=str,
                        help="elasticsearchのport番号, e.g. 9200", dest="port")

    args = parser.parse_args()
    # print(args)
    LOGGER.info(args)

    s2v = Struc2Vec(args)
    if args.mode == "regi":
        s2v.register_dataset()
    elif args.mode == "vec":
        s2v.convert_vector()
    elif args.mode == "save":
        s2v.save_result()
    elif args.mode == "plot":
        pass
    # if args.statistics_learning_data:
    #     frequency(
    #         index=args.data_index,
    #         host=args.host,
    #         port=args.port,
    #         path="freq.csv",
    #     )
    # else:
    #     if args.dir:
    #         s2v.register_dataset()
    #     # s2v.convert_vector()
    #     # if not args.not_save:
    #     #     s2v.save_result()

class Struc2Vec(object):
    """struc2vecのメインクラス
    """

    def __init__(self, args):
        self.args = args
        if self.args.fs:
            self.args.fs = list(map(int, self.args.fs.split(",")))
        LOGGER.info("struc2vec start")
        # terminal_to_rootは，巨大データになりやすいため，小規模コーパスのみ
        # self.ftype_list = ["root_to_terminal","path_length","terminal_to_root"]
        # self.ftype_list = ["root_to_terminal","path_length"]
        self.ftype_list = [self.args.ftype]

    def register_dataset(self):
        """register_dataset
        ・入力されるデータセットを，elasticsearchに登録する
        ・svm_perfで実行できる形に前処理する．

        戦略:
            parallel処理じゃないと，そもそもだめ．遅すぎる
            1. (parallel)1000単位のファイル(java, あるいはxml)を対象
            2. (parallel)java -> xml -> pre-learning_data (tmp_workspaceというインデックスにxmlをぶっこむ)
            3. (単体)pre-learning_dataをsearchし，set.union+辞書化でfeature_dict, symbol_dictを作成&登録
        """

        # init
        LOGGER.info("register dataset > init")
        work_root = Workspace(directory="tmp_workspace")
        num_core = 1  # core
        if self.args.parallel:
            num_core = -1

        # data mapping
        LOGGER.info("register dataset > mapping %s", self.args.data_index)
        ins_mapping = MappingELS(index=self.args.data_index, host=self.args.host, port=self.args.port)
        ins_mapping.data_mapping2()
        ins_mapping.register()

        # parallel
        LOGGER.info("register dataset > elasticsearch instance")
        els_sample = RegisterELS(index=self.args.data_index, doc_type="sample", host=self.args.host, port=self.args.port)
        ins_common_sample_loop = CommonSampleLoop(root_dir=self.args.dir, ext=self.args.ext, index=self.args.data_index, host=self.args.host, port=self.args.port)

        LOGGER.info("register dataset > register input data")
        for sample_list in ins_common_sample_loop.yield_sample():
            Parallel(n_jobs=num_core, verbose=1, backend="threading")([
                delayed(register_sample)(
                    sample=sample,
                    args=self.args,
                    work_root_path=work_root.get_path(),
                    ins_sample=els_sample,
                ) for sample in sample_list])
            els_sample.register()  # 登録
        # els_sample.register(wait=True, mes="sample")  # ちょい待ち
            # work_root.refresh() # Workspaceのリフレッシュ

        # register features
        # ins_symbol = Symbol()
        for ftype in self.ftype_list:
            # tmp_workspace mapping
            LOGGER.info("register dataset > mapping tmp_workspace")
            ins_tmp_mapping = MappingELS(index="tmp_workspace", host=self.args.host, port=self.args.port)
            ins_tmp_mapping.tmp_workspace_mapping()
            ins_tmp_mapping.register()
            els_tmp_workspace = RegisterELS(index="tmp_workspace", doc_type="tmp_workspace", host=self.args.host, port=self.args.port)

            # parallel
            LOGGER.info("register dataset > start ftype: {}".format(ftype))
            for sample_list in ins_common_sample_loop.yield_xml():
                Parallel(n_jobs=num_core, verbose=1, backend="threading")([
                    delayed(register_feature)(
                        length_feature_path=self.args.path,
                        sample=sample,
                        ftype=ftype,
                        ins_tmp_workspace=els_tmp_workspace,
                        identifier=self.args.identifier,
                    ) for sample in sample_list])
                els_tmp_workspace.register()  # 登録
            # els_tmp_workspace.register(wait=True, mes="tmp_workspace")  # ちょい待ち
                # work_root.refresh() # Workspaceのリフレッシュ

            # update feature
            LOGGER.info("register dataset > register ftype: {}".format(ftype))
            ins_feature = Feature(
                index=self.args.data_index,
                host=self.args.host,
                port=self.args.port,
                ftype=ftype,
                num_core=num_core,
            )
            ins_feature.update_learning_data(ftype)
            LOGGER.info("register dataset > end ftype: {}".format(ftype))

        # create feature & symbol
        # LOGGER.info("register dataset > register feature")
        # ins_feature.register_feature()
        # LOGGER.info("register dataset > register symbol")
        # ins_symbol.register_symbol(index=self.args.data_index, host=self.args.host, port=self.args.port)

        # end
        work_root.delete()
        LOGGER.info("register dataset > end")

    def convert_vector(self):
        """vector表現に変換
        ・推論モデルを使用し，メソッドの数値表現を獲得
        ・elasticsearchとファイルにSVMスコアを出力
        """

        # struc2vec mapping
        LOGGER.info("convert vector > init")
        LOGGER.info("convert vector > mapping vector")
        ins_mapping = MappingELS(index=self.args.s2v_index, host=self.args.host, port=self.args.port)
        # ins_mapping.experiment_mapping()
        ins_mapping.s2v_mapping()
        ins_mapping.register()

        # init
        LOGGER.info("convert vector > init")
        num_core = 1
        if self.args.parallel:
            num_core = -1
        work_root = Workspace(directory="tmp_workspace")
        ins_common_id = CommonId(host=self.args.host, port=self.args.port, index=self.args.data_index, ftype_list=self.ftype_list)
        pre_class_score = ClassScoreDoc(index=self.args.s2v_index, host=self.args.host, port=self.args.port,)
        pre_class_score.register_class_score(class_id_list=ins_common_id.get_class_id_list(),ftype_list=self.ftype_list,)

        # infer model
        if self.args.model == "svm":
            for ftype in self.ftype_list:
                ins_calc_svm = CalcSVM(
                    args=self.args,
                    ftype=ftype,
                    num_core=num_core,
                    work_root=work_root,
                    ins_common_id=ins_common_id,
                )
                ins_calc_svm.run()
        elif self.args.model == "nn":
            for ftype in self.ftype_list:
                ins_calc_nn = CalcNN(
                        args=self.args,
                        ftype=ftype,
                        num_core=num_core,
                        work_root=work_root,
                        ins_common_id=ins_common_id,
                    )
                ins_calc_nn.run()
        elif self.args.model == "rnn":
            pass

        # end
        work_root.delete()
        LOGGER.info("convert vector > end")

    def save_result(self):
        """推論結果のbz2による圧縮ファイルの保存
        """

        LOGGER.info("save result start")

        work_root = Workspace(directory="tmp_workspace")
        ins_save_result = SaveResult_SVM(
            s2v_index=self.args.s2v_index,
            data_index=self.args.data_index,
            host=self.args.host,
            port=self.args.port,
            ftype=self.args.ftype,
        )
        for num_fs in [1,2,3,4,5,99999]:
            LOGGER.info("evaluate_score_fs{}".format(num_fs))
            ins_save_result.evaluate_score(num_fs)
        LOGGER.info("evaluate_score_max")
        ins_save_result.evaluate_score_max()


        LOGGER.info("dis_rep_class")
        ins_save_result.dis_rep_class(work_root.get_path())

        LOGGER.info("plot program pattern")
        ins_save_result.plot_program_pattern()

        LOGGER.info("search_sorce_code")
        ins_save_result.search_sorce_code(work_root.get_path())

        # LOGGER.info("dis_rep_sample")
        # ins_save_result.dis_rep_sample()

        # delete
        work_root.delete()
        LOGGER.info("save result end")

if __name__ == "__main__":
    main()
