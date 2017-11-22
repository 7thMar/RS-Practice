#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import profile
from pylab import *
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as f
import random
import UserCF as UserCF
import ItemCF as ItemCF
cnfont = f.FontProperties(
    fname='/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc')
matplotlib.rcParams['axes.unicode_minus'] = False

# plt.title('中文示例', fontproperties=cnfont)


# 读取数据
def readData(path, sep=','):
    return pd.read_csv(path, sep=sep, usecols=[0, 1, 2])


# 分成训练集和测试集
def SplitData(data, M, seed):
    test = {}
    train = {}
    # { userID: [item list][rating list] }
    count = data.shape[0]  # 数据总数
    np.random.seed(seed)
    index = np.arange(count)
    test_index = np.random.choice(index, int(count / M), replace=False)
    train_index = np.setdiff1d(index, test_index)

    data_train = data.iloc[train_index, :]
    train_group = data_train.groupby("UserID")["MovieID", "Rating"]
    for k, v in train_group:
        train[k] = v.as_matrix(["MovieID", "Rating"]).T

    data_test = data.iloc[test_index, :]
    test_group = data_test.groupby("UserID")["MovieID", "Rating"]
    for k, v in test_group:
        test[k] = v.as_matrix(["MovieID", "Rating"]).T

    return train, test


# N: 物品数目
# train: {user: item, }
# 和推荐数目N有很大的关系
# 准确率 召回率 覆盖率 流行度
def PrecisionAndRecallAndCoverageAndPopularity(train, test, item_popularity, K, W, N, method=1):
    hit = 0
    num_rank = 0
    num_tu = 0
    recommend_items = set()
    all_items = set()
    popularity = 0.0
    if method == 1:
        for user in train:  # test / train
            if user not in test:
                continue
            all_items = all_items | set(train[user][0])
            tu = test[user][0]
            rank = UserCF.Recommend(user, train, W, K, N)
            recommend_items = recommend_items | set(rank)
            #  hit += len(np.intersect1d(rank, tu))
            for item in rank:
                if item in tu:
                    hit += 1
                popularity += math.log(1 + item_popularity[item])
            #  for item, value in rank.items():
            #  if item in tu:
            #  hit += 1
            num_rank += len(rank)
            num_tu += len(tu)
        #  print('Hit: ', hit)
        #  print('Rank num: ', num_rank)
        #  print('Test user\'s item num:', num_tu)
        #  print(len(all_items), len(recommend_items))
        return hit / (num_rank * 1.0), hit / (
            num_tu * 1.0), len(recommend_items) / (len(all_items) * 1.0), popularity / (num_rank * 1.0)
    elif method == 2:
        for user in train:  # test / train
            if user not in test:
                continue
            all_items = all_items | set(train[user][0])
            tu = test[user][0]
            rank = ItemCF.Recommend(user, train, W, K, N)
            for item, _ in rank:
                if item in tu:
                    hit += 1
                popularity += math.log(1 + item_popularity[item])
                recommend_items.add(item)
            num_rank += len(rank)
            num_tu += len(tu)
        #  print('Hit: ', hit)
        #  print('Rank num: ', num_rank)
        #  print('Test user\'s item num:', num_tu)
        #  print(len(all_items), len(recommend_items))
        return hit / (num_rank * 1.0), hit / (
            num_tu * 1.0), len(recommend_items) / (len(all_items) * 1.0), popularity / (num_rank * 1.0)


# 获取商品的流行度
def ItemsPopularity(data, M, seed):
    count = data.shape[0]  # 数据总数
    np.random.seed(seed)
    index = np.arange(count)
    test_index = np.random.choice(index, int(count / M), replace=False)
    train_index = np.setdiff1d(index, test_index)

    data_train = data.iloc[train_index, :]
    item_popularity = data_train.groupby("MovieID")["UserID"].count().to_dict()

    return item_popularity


def MostPopularResult(train, test, item_popularity, N):
    hit = 0
    num_rank = 0
    num_tu = 0
    recommend_items = set()
    all_items = set()
    popularity = 0.0
    popularity_order = sorted(item_popularity.items(), key=lambda d: d[1], reverse=True)
    for user in train:  # test / train
        if user not in test:
            continue
        all_items = all_items | set(train[user][0])
        tu = test[user][0]
        n = 0
        for item, _ in popularity_order:
            if item in train[user][0]:
                continue
            if item in tu:
                hit += 1
            popularity += math.log(1 + item_popularity[item])
            recommend_items.add(item)
            n += 1
            if n == N:
                break
        num_rank += N
        num_tu += len(tu)
    return hit / (num_rank * 1.0), hit / (
        num_tu * 1.0), len(recommend_items) / (len(all_items) * 1.0), popularity / (num_rank * 1.0)


def RandomResult(train, test, item_popularity, N):
    hit = 0
    num_rank = 0
    num_tu = 0
    recommend_items = set()
    all_items = set()
    popularity = 0.0
    for user in train:  # test / train
        if user not in test:
            continue
        all_items = all_items | set(train[user][0])
    all_items = list(all_items)
    for user in test:
        tu = test[user][0]
        n = 0
        while n < N:
            item = all_items[np.random.randint(len(all_items))]
            if item in train[user][0]:
                continue
            if item in tu:
                hit += 1
            recommend_items.add(item)
            popularity += math.log(1 + item_popularity[item])

            n += 1
        num_rank += N
        num_tu += len(tu)
    return hit / (num_rank * 1.0), hit / (
        num_tu * 1.0), len(recommend_items) / (len(all_items) * 1.0), popularity / (num_rank * 1.0)
