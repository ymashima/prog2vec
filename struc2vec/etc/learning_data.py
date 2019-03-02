# coding:utf-8

"""learning_dataに関しての統計情報
"""

import os
from logging import getLogger

from struc2vec.utils.log_setting import set_log
from struc2vec.ELS.read import ReadELS
from struc2vec.utils.file_control import write_file

LOGGER = getLogger(__name__)
set_log(LOGGER)

def frequency(index, host, port, path):
    ins_sample = ReadELS(index=index, host=host, port=port, doc_type="sample")
    ftype_dict = {}
    LOGGER.info("frequency > read sample")
    ftype_list = ["path_length", "root_to_terminal"]
    for ftype in ftype_list:
        ftype_dict[ftype] = []
        feature_id_dict = {}
        for i,sample in enumerate(ins_sample.search_records(column=ftype),start=1):
            # class_sample_id = sample["_id"]
            sample_data = sample["_source"][ftype]
            if i % 10000 == 0:
                LOGGER.info("sample:%d", i)
            for k in list(map(int, sample_data.split(" "))):
                if k not in feature_id_dict:
                    feature_id_dict[k] = 1
                else:
                    feature_id_dict[k] += 1
        ftype_dict[ftype] = sorted(feature_id_dict.items(), key=lambda x:x[1], reverse=True)
    import pdb; pdb.set_trace()

    # write
    # LOGGER.info("frequency > write data")
    # source = "feature_id,count{}".format(os.linesep)
    # for i, (k, v) in enumerate(feature_id_list, start=1):
    #     if i % 10000 == 0:
    #         LOGGER.info("sample:%d", i)
    #     source += "{},{}{}".format(k, v, os.linesep)
    # write_file(path=path, source=source)
