# coding:utf-8

"""logの設定
"""

from datetime import datetime
from logging import basicConfig, getLogger, config, DEBUG, INFO, StreamHandler, FileHandler, Formatter

# global log file name
log_filename = "log/{}_s2v.log".format(datetime.now().strftime("%Y%m%d_%H%M%S"))

def set_log(LOGGER, root=False):
    """logの設定

    Args:
        LOGGER (getLogger): getLoggerオブジェクトからの各ファイル名
    """

    LOGGER.setLevel(INFO)
    # FORMAT="%(asctime)s %(levelname)s [in %(pathname)s]: >>> %(message)s"
    FORMAT="%(asctime)s %(levelname)s >>> %(message)s >>> [in %(name)s]:"
    # Stream
    handle1 = StreamHandler()
    handle1.setFormatter(Formatter(FORMAT))
    LOGGER.addHandler(handle1)
    # File
    write_mode = "a"
    if root:
        write_mode = "w"
    handle2 = FileHandler(
        filename=log_filename,
        mode=write_mode,
        encoding="utf-8"
    )
    handle2.setLevel(INFO)
    handle2.setFormatter(Formatter(FORMAT))
    LOGGER.addHandler(handle2)
