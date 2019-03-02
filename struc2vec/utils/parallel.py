# coding:utf-8

"""joblibのparallelの設定,まだだめ
"""

from logging import getLogger
from joblib import Parallel

from struc2vec.utils.log_setting import set_log

LOGGER = getLogger(__name__)
set_log(LOGGER)

class Parallel_log(Parallel):
    def __init__(
        self,
        n_jobs=None,
        backend=None,
        verbose=0,
        timeout=None,
        pre_dispatch='2 * n_jobs',
        batch_size='auto',
        temp_folder=None,
        max_nbytes='1M',
        mmap_mode='r',
        prefer=None,
        require=None
    ):
        super().__init__(
            n_jobs=None,
            backend=None,
            verbose=0,
            timeout=None,
            pre_dispatch='2 * n_jobs',
            batch_size='auto',
            temp_folder=None,
            max_nbytes='1M',
            mmap_mode='r',
            prefer=None,
            require=None
        )
    def _print(self, msg, msg_args):
        """Display the message on stout or stderr depending on verbosity
        _printメソッドのオーバーライド
        """
        if not self.verbose:
            return
        msg = msg % msg_args
        LOGGER.info('[%s]: %s\n' % (self, msg))
        import pdb; pdb.set_trace()