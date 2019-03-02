# coding:utf-8

"""elasticsearchデータのバックアップ
"""

import os
from datetime import datetime

from struc2vec.ELS.read import ReadELS
from struc2vec.utils.file_control import Bz2file

class ElasticsearchData(object):
    def __init__(self, args):
        self.els_sample_read = ReadELS(doc_type="sample", host=args.index, port=args.port, index=args.index,)
        self.els_PL_read = ReadELS(doc_type="path_length", host=args.index, port=args.port, index=args.index,)
        self.els_PL_symbol_read = ReadELS(doc_type="path_length_symbol", host=args.index, port=args.port, index=args.index,)
        self.els_RtT_read = ReadELS(doc_type="root_to_terminal", host=args.index, port=args.port, index=args.index,)
        self.els_RtT_symbol_read = ReadELS(doc_type="root_to_terminal_symbol", host=args.index, port=args.port, index=args.index,)

    def backup(self, path):
        Bz2file(os.path.join(path, "{}_sample.bz2".format(datetime.now().strftime("%Y%m%d_%H%M%S"))))

        for record in self.els_sample_read.search_records(
            column="class_id,class_name,sample_id,java,xml,dot,path_length,root_to_terminal,terminal_to_root,lines_of_code",
        ):
            pass