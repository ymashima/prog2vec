# coding:utf-8

"""xml解析
"""

from xml.parsers.expat import ParserCreate


class PathLength(object):
    """インスタンス化し，run_parse()を実行することでxmlを解析

    Args:
        xml (str): xmlファイルを読み込んだソース
        length_feature_path (str): ASTを区切るpathの長さ
    """

    def __init__(self, xml, length_feature_path, identifier=True):
        """

        Vars:
            # 解析に使用するデータ構造
            self.stack (list): 出現したfeatureの一時保管リスト
            self.feture_set (set): 解析したxmlファイル上で出現するfeature集合
            self.length_feature_path (int):  ASTを区切るpathの長さ
            self.xml (str): xmlファイルを読み込んだソース

            # 解析の設定
            self.parse_xml (ParserCreate): ParserCreateのインスタンス
        """

        self.stack = []
        self.feature_set = set() # 出現する非終端記号と終端記号
        self.length_feature_path = int(length_feature_path)
        self.xml = xml
        self.identifier = identifier  # 終端記号を含めるかどうか

        # parser setting
        self.parse_xml = ParserCreate()
        self.parse_xml.buffer_text = True
        self.parse_xml.StartElementHandler = self.start_element  # 開始タグの動作
        self.parse_xml.EndElementHandler = self.end_element  # 終了タグの動作

    def start_element(self, name, attrs):
        """xml開始タグに対する操作
        順次stackにfeatureを格納し，パスの長さに合致するfeatureを，
        feature_setに格納する．

        Args:
            name (str): xmlタグの名前
            attrs (dict): xmlタグに付随する属性
        """

        if self.identifier is True or name != "identifier":
            elem_name = "{}:{}".format(attrs['name'], name)
            self.stack.append(elem_name)
            feature_set = self.get_feature_set()
            self.feature_set |= feature_set
        # import pdb; pdb.set_trace()

    def get_feature_set(self):
        """指定したpath以下のfeatureを取得

        Vars:
            feature_set (set): path以下のfeature集合

        Returns:
            (set): path以下のfeature集合
        """

        feature_set = set()
        l = self.length_feature_path
        feature_set = set(['/'.join(self.stack[-i:]) for i in range(1,l+1,1)])
        return feature_set

    def end_element(self, name):
        """xml終了タグに対する操作

        Args:
            name (str): xmlタグの名前
        """

        if self.identifier is True or name != "identifier":
            self.stack.pop()
        # import pdb; pdb.set_trace()

    def run_parse(self):
        """xmlの解析

        Returns:
            self.feature_set (set): xmlファイルに出現したfeature集合
            self.base_feature_dict (dict): 更新されたfeture_base辞書
        """

        self.parse_xml.Parse(self.xml)  # 解析の実行

    def get_feature_str(self):
        """feature_setを文字列として引っ付けて戻す

        Returns:
            (str): feature_setの文字列
        """

        feature_str = " ".join(self.feature_set)
        return feature_str


class RootToTerminal(object):
    """インスタンス化し，run_parse()を実行することでxmlを解析
    ASTのRootからTerminalまでの系列の部分集合を取得

    Args:
        xml (str): xmlファイルを読み込んだソース
    """

    def __init__(self, xml, identifier):
        """

        Vars:
            # 解析に使用するデータ構造
            self.stack (list): 出現したfeatureの一時保管リスト
            self.feture_set (set): 解析したxmlファイル上で出現するfeature集合
            self.xml (str): xmlファイルを読み込んだソース

            # 解析の設定
            self.parse_xml (ParserCreate): ParserCreateのインスタンス
        """

        self.stack = []
        self.feature_set = set() # 出現する非終端記号と終端記号
        self.xml = xml
        self.identifier = identifier

        # parser setting
        self.parse_xml = ParserCreate()
        self.parse_xml.buffer_text = True
        self.parse_xml.StartElementHandler = self.start_element  # 開始タグの動作
        self.parse_xml.EndElementHandler = self.end_element  # 終了タグの動作

    def start_element(self, name, attrs):
        """xml開始タグに対する操作
        順次stackにfeatureを格納

        Args:
            name (str): xmlタグの名前
            attrs (dict): xmlタグに付随する属性
        """

        if self.identifier is True or name != "identifier":
            elem_name = "{}:{}".format(attrs['name'], name)
            self.stack.append(elem_name)
        # import pdb; pdb.set_trace()

    def get_feature_set(self):
        """ASTのRootからTerminalまでのfeatureを取得

        Vars:
            feature_set (set): RootからTerminalまでのfeature集合

        Returns:
            (set): feature集合
        """

        feature_set = set()
        feature_set = set(['/'.join(self.stack[:l]) for l in range(1, len(self.stack) + 1)])
        # import pdb; pdb.set_trace()
        return feature_set

    def end_element(self, name):
        """xml終了タグに対する操作
        feature_setにRootからTerminlaまでのfeatureを格納

        Args:
            name (str): xmlタグの名前
        """

        if self.identifier is True or name != "identifier":
            feature_set = self.get_feature_set()
            self.feature_set |= feature_set
            self.stack.pop()
        # import pdb; pdb.set_trace()

    def run_parse(self):
        """xmlの解析

        Returns:
            self.feature_set (set): xmlファイルに出現したfeature集合
            self.base_feature_dict (dict): 更新されたfeture_base辞書
        """

        self.parse_xml.Parse(self.xml)  # 解析の実行

    def get_feature_str(self):
        """feature_setを文字列として引っ付けて戻す

        Returns:
            (str): feature_setの文字列
        """

        feature_str = " ".join(self.feature_set)
        return feature_str

'''
class TerminalToRoot(object):
    """インスタンス化し，run_parse()を実行することでxmlを解析
    ASTのTerminalからRootまでの系列の部分集合を取得

    Args:
        xml (str): xmlファイルを読み込んだソース
    """

    def __init__(self, xml):
        """

        Vars:
            # 解析に使用するデータ構造
            self.stack (list): 出現したfeatureの一時保管リスト
            self.feture_set (set): 解析したxmlファイル上で出現するfeature集合
            self.xml (str): xmlファイルを読み込んだソース

            # 解析の設定
            self.parse_xml (ParserCreate): ParserCreateのインスタンス
        """

        self.stack = []
        self.feature_set = set() # 出現する非終端記号と終端記号
        self.xml = xml

        # parser setting
        self.parse_xml = ParserCreate()
        self.parse_xml.buffer_text = True
        self.parse_xml.StartElementHandler = self.start_element  # 開始タグの動作
        self.parse_xml.EndElementHandler = self.end_element  # 終了タグの動作

    def start_element(self, name, attrs):
        """xml開始タグに対する操作
        順次stackにfeatureを格納

        Args:
            name (str): xmlタグの名前
            attrs (dict): xmlタグに付随する属性
        """

        elem_name = "{}:{}".format(attrs['name'], name)
        self.stack.append(elem_name)
        # import pdb; pdb.set_trace()

    def get_feature_set(self):
        """ASTのTerminalからRootまでのfeatureを取得

        Vars:
            feature_set (set): TerminalからRootまでのfeature集合(必ずterminalを含む)

        Returns:
            (set): feature集合
        """

        feature_set = set()
        symbol_type = self.stack[-1].split(":")[-1]
        if symbol_type == "identifier":
            feature_set = set(['/'.join(self.stack[-1 * l:]) for l in range(1, len(self.stack) + 1)])
            # import pdb; pdb.set_trace()
        return feature_set

    def end_element(self, name):
        """xml終了タグに対する操作
        feature_setにTerminalからRootまでのfeatureを格納

        Args:
            name (str): xmlタグの名前
        """

        feature_set = self.get_feature_set()
        self.feature_set |= feature_set
        self.stack.pop()

    def run_parse(self):
        """xmlの解析

        Returns:
            self.feature_set (set): xmlファイルに出現したfeature集合
            self.base_feature_dict (dict): 更新されたfeture_base辞書
        """

        self.parse_xml.Parse(self.xml)  # 解析の実行

    def get_feature_str(self):
        """feature_setを文字列として引っ付けて戻す

        Returns:
            (str): feature_setの文字列
        """

        feature_str = " ".join(self.feature_set)
        return feature_str
'''
