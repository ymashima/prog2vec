# coding:utf-8

"""elasticsearchを書き込みで使用するモジュール
"""

from logging import getLogger
import time

from elasticsearch import Elasticsearch
from elasticsearch import helpers
from struc2vec.utils.log_setting import set_log

LOGGER = getLogger(__name__)
set_log(LOGGER)

class RegisterELS(object):
    """elasticsearchへ書き込む

    Args:
        host (str): elasticsearchのhost
        port (str): elasticsearchのpost
        index_name (str): インデックス名
        doc_type (str): ドキュメント名

    """

    def __init__(self, host, port, index, doc_type):
        """

        Vars:
            self.els (Elasticsearch): elasticsearchのインスタンス
            self.index_list (list): レコードを登録する一時保管リスト
            self.index (str): インデックス名
            self.doc_type (str): ドキュメント名
        """

        self.els = Elasticsearch(host=host, port=port, timeout=1000)
        self.els.cluster.health(params={
                "wait_for_status": "yellow",
                "request_timeout": 1000,
            })
        self.insert_list = []
        self.index = index
        self.doc_type = doc_type

    def create(self, record_id, dict_source):
        """elasticsearchに登録するdictオブジェクトを作成し，
        insert_listに格納する

        Args:
            index_name (str): インデックス名
            doc_type (str): ドキュメント名
            record_id (str): レコードのid
            dict_source (dict): レコードのカラムのデータ
        """

        dict_register = {
            '_op_type': "create",
            '_index': self.index,
            '_type': self.doc_type,
            '_id': record_id,
            '_source': dict_source,
        }
        self.insert_list.append(dict_register)

    def update(self, record_id, dict_source):
        """elasticsearchに登録するdictオブジェクトを作成し，
        insert_listに格納する

        Args:
            index_name (str): インデックス名
            doc_type (str): ドキュメント名
            record_id (str): レコードのid
            dict_source (dict): レコードのカラムのデータ
        """

        dict_register = {
            '_op_type': "update",
            '_index': self.index,
            '_type': self.doc_type,
            '_id': record_id,
            'doc': dict_source,
        }
        self.insert_list.append(dict_register)

    def register(self, wait=True, mes=""):
        """elasticsearchに登録
        """

        helpers.bulk(self.els, self.insert_list)
        # if wait:
        LOGGER.info("wait register {} ...".format(mes))
        time.sleep(2)  # 登録後に少し待つ必要がある...なぜ？
        self.insert_list = []
