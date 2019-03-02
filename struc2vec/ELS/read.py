# coding:utf-8

"""
elasticsearchを読み込み専用で使用するモジュール
"""

from elasticsearch import Elasticsearch


class ReadELS(object):
    """elasticsearhcの読み込みクラス

    Args:
        host (str): ホスト名
        port (str): ポート名
        index_name (str): インデックス名
        doc_type (str): ドキュメント名
    """

    def __init__(self, host, port, index, doc_type):
        """

        Vars:
            self.els (Elasticsearch): elasticsearchのインスタンス
            self.index (str): インデックス名
            self.doc_type (str): ドキュメント名
        """

        self.els = Elasticsearch(host=host, port=port, timeout=1000)
        self.els.cluster.health(params={
            "wait_for_status": "yellow",
            "request_timeout": 1000,
        })
        self.index = index
        self.doc_type = doc_type

    def search_records(self, column, body="", sort="", size=10000):
        """elasticsearchの検索結果を，1件ずつジェネレータで返す

        Args:
            column (str): 取得するカラムを指定,コンマで繋げる
            body (dict, optional): デフォルトはNone．詳細な検索内容を指定
            sort (str, optional): デフォルトは空文字．カラムによる並び替えを指定

        Yield:
            検索結果の1件をdict型で生成（ジェネレータ）
        """

        search_result = self.els.search(
            index=self.index,
            doc_type=self.doc_type,
            body=body,
            params={
                "scroll": "25m",
                "_source_include": column,
                "sort": sort,
                "size": size,
            }
        )

        while search_result['hits']['hits']:
            scroll_id = search_result['_scroll_id']
            search_result = search_result['hits']['hits']
            for result_item in search_result:
                yield result_item   # 検索結果を1件ずつ返す
            search_result = self.els.scroll(
                scroll_id=scroll_id,
                params={
                    "scroll": "25m",
                }
            )

    def search_count(self, body=""):
        """ドキュメント名のドキュメント数の取得

        Return:
            (int): ドキュメント数
        """

        search_result = self.els.search(
            index=self.index,
            doc_type=self.doc_type,
            body=body,
            params={
                "_source_include": "",
                "size": 0,
            }
        )

        hit_count = search_result["hits"]["total"]
        return hit_count

    def search_top_hits(self, column, group, size=1, sort_dict={}):
        """elasticsearchの検索結果を，groupごとに取得

        Args:
            column (str): 取得するカラムを指定,コンマで繋げる
            group (str): 集約するカラムを指定
            size (int, optional): デフォルトは1．集約後の検索件数の上限
            sort_dict (dict, optional): デフォルトは空辞書．カラムによる並び替えを指定
                example: sort_dict = {
                    "f1": {"order": "desc"},
                    "feature_selection": {"order": "asc"},
                }

        return:
            検索結果のリストをdict型で生成
        """

        body = {
            "size": 0,
            "aggs": {self.doc_type: {
                "terms": {
                    "field": group,
                    "size": 10000 # 集約の種類数
                },
                "aggs": {group: {"top_hits": {
                    "_source": {"includes": column.split(",")},
                    "size": size, # 各集約から取得するドキュメント数
                }}}
            }}
        }
        if sort_dict:
            body["aggs"][self.doc_type]["aggs"][group]["top_hits"]["sort"] = [sort_dict]

        search_result = self.els.search(
            index=self.index,
            doc_type=self.doc_type,
            body=body,
            params={
            }
        )

        search_result = search_result['aggregations'][self.doc_type]['buckets']
        return search_result

    def get_record(self, doc_id, column=""):
        """idに合致したrecordを1件取得

        Args:
            doc_id (str): ドキュメントID
            column (str, optional): デフォルトは空文字,取得したいカラム

        Returns:
            get_result (dict): 検索結果
        """

        get_result = self.els.get_source(
            index=self.index,
            doc_type=self.doc_type,
            id=doc_id,
            params={
                "_source_include": column,
            }
        )

        return get_result
