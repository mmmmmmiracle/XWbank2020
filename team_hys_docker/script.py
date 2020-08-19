#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from tensorflow import keras
from utils import *
from models import *
from preprocess import *

# pl
train_data, train_labels, test_data = get_train_test_data(pseudo_labels_file=
                                                'data/pl.csv')
# 'mixup', 'mixup2', 'mixup3', 'mixup4', 'mixup5', 'mixup6', 'reverse'
# set_data_enhance(['noise', 'mixup', 'mixup2', 'mixup3', 'mixup4', 'mixup5', 'mixup6', 'reverse'])


# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


# # 转换成float32节省显存，以及one_hot编码
# num_classes=19
# train_data = tf.cast(train_data, tf.float32).numpy()
# train_labels = tf.one_hot(train_labels, num_classes).numpy()


# 训练
# histories, evals = kfcv_fit(builder=lambda : ComplexConv1D(train_data.shape, 19),
#                                 x=train_data, y=train_labels,
#                                 epochs=30,
#                                 checkpoint_path = './models/conv1d/',
#                                 batch_size=128
#                                 )
# 推断
infer('conv1d', get_test_data(), 'submission.csv')






