#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.font_manager as f
cnfont = f.FontProperties(
    fname='/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc')
matplotlib.rcParams['axes.unicode_minus'] = False

# plt.title('中文示例', fontproperties=cnfont)


class ContentItemKNN(object):

    def __init__(self):
        pass

    def ReadData(self, path, sep):
        data = pd.read_csv(path, sep=sep)
        pass

    def SplitData(self, data, m, seed):
        pass

    def TF_IDF(self):
        pass

    def CalculateSimilarity(self):
        # 1. calc invert table
        # 2. calc similarity
        pass

    # precision recall coverage popularirt
    def evaluation(train, test, item_popularity, W, K, N):
        hit = 0
        rank_num = 0
        tu_num = 0
        recommend_items = set()
        popularity = 0.0
        for user in train:  # test / train
            if user not in test:
                continue
            tu = test[user]  # test: {user: items}
            rank = Recommend(user, train, W, K, N)
            recommend_items = recommend_items | set(rank)
            for item in rank:
                if item in tu:
                    hit += 1
                popularity += math.log(1 + item_popularity[item])
            rank_num += len(rank)
            tu_num += len(tu)
        return hit / (rank_num * 1.0), hit / (
            tu_num * 1.0), len(recommend_items) / (
                len(all_items) * 1.0), popularity / (rank_num * 1.0)
