# coding:utf-8

"""neural networkに関するモデル
"""

# import
import numpy as np
import os

# chainer
import chainer
from chainer import Function, gradient_check, report, training, utils, Variable, initializers
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer.training import extensions
from chainer.backends.cuda import to_cpu
import chainer.functions as F
import chainer.links as L

import matplotlib.pyplot as plt
from chainer.datasets import mnist
from chainer.dataset import concat_examples

# chainer config
chainer.global_config.autotune = True
chainer.global_config.dtype = np.dtype("float16")
chainer.cuda.set_max_workspace_size(512*1024*1024)  # gpu
chainer.cudnn_fast_batch_normalization = True  # gpu

class NNModel(Chain):
    """neural networkのネットワーク定義

    Args:
        Chain (Chain): NNの継承モデル
        n_id (int): 入力層のユニット数
        n_mid (int): 隠れ層のユニット数
        n_out (int): 出力層のユニット数
    """

    def __init__(self, n_id, n_mid, n_out):
        initializer = initializers.Normal()
        super(NNModel, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_id, n_mid, initialW=initializer, initial_bias=0)
            self.l2 = L.Linear(n_mid, n_out, initialW=initializer, initial_bias=0)

    def __call__(self, x):
        """neural networkのforward計算

        Args:
            x (ndarray): numpy形式の入力

        Returns:
            (ndarray): 出力結果
        """

        self.h1 = F.relu(self.l1(x))
        self.out = self.l2(self.h1)
        return self.out

    def get_hidden_layer(self, x):
        """中間層出力

        Args:
            x (ndarray): numpy形式の入力

        Returns:
            (ndarray): 中間層出力
        """

        self.h1 = F.relu(self.l1(x))
        return self.h1

