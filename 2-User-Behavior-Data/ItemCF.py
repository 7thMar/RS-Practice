#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: ItemCF.py
Author: K
Email: 7thmar37@gmail.com
Github: https://github.com/7thmar
Description: ItemCF
"""

import math
import numpy as np
import pandas as pd
import time


def ItemSimilarityVersion1(train, N, items_user):
    """ ItemCF 算法 计算物品相似度

    Desc:


    Args:


    Returns:


    """
    start = time.time()
    items_list = list(items_user.keys())
    W = dict()
    for index, i in enumerate(items_list):
        if i not in W:
            W[i] = dict()
        for j in items_list[index + 1:]:
            if j not in W:
                W[j] = dict()
            W[i][j] = len(items_user[i].intersection(items_user[j])) / math.sqrt(N[i] * N[j])
            W[j][i] = W[i][j]

    W = pd.DataFrame(W)
    #  print(W)
    end = time.time()
    print('Cacl Similarity: ', end - start, 's')

    return W


def ItemSimilarityVersion2(train, N, items_user):
    """ ItemCF IUF 计算物品相似度

    Desc:


    Args:


    Returns:


    """
    start = time.time()

    items_list = list(items_user.keys())
    #  ignore_limit = len(items_list) * 0.5
    W = dict()
    for index, i in enumerate(items_list):
        if i not in W:
            W[i] = dict()
        for j in items_list[index + 1:]:
            if j not in W:
                W[j] = dict()
            n = 0
            users = items_user[i].intersection(items_user[j])
            for k in users:
                nu = len(train[k][0])
                #  if nu > ignore_limit:
                    #  print(k, 'ignore')
                    #  continue
                n += 1 / math.log(1 + nu * 1.0)
            W[i][j] = n / math.sqrt(N[i] * N[j])
            W[j][i] = W[i][j]

    W = pd.DataFrame(W)
    #  print(W)
    end = time.time()
    print('Cacl Similarity: ', end - start, 's')

    return W


def ItemSimilarityNorm(train, N, items_user):
    """ ItemCF Norm 计算物品相似度

    Desc:


    Args:


    Returns:


    """
    start = time.time()
    items_list = list(items_user.keys())
    W = dict()
    for index, i in enumerate(items_list):
        if i not in W:
            W[i] = dict()
        for j in items_list[index + 1:]:
            if j not in W:
                W[j] = dict()
            W[i][j] = len(items_user[i].intersection(items_user[j])) / math.sqrt(N[i] * N[j])
            W[j][i] = W[i][j]
        #  maxium = max(W[i].values())
        #  for j in items_list[index + 1:]:
            #  W[i][j] /= maxium
            #  W[j][i] = W[i][j]

    W = pd.DataFrame(W)
    W = W.fillna(0)
    for i in W.index:
        maxium = max(W.loc[i])
        W.loc[i] /= maxium
        #  W.loc[i, i:] /= maxium
        #  W.loc[i, :i] = W.loc[i - 1, :i]

    #  W.loc[1].to_excel('Item-Similarity-Norm-1.xlsx', 'Sheet1')
    end = time.time()
    print('Cacl Similarity: ', end - start, 's')
    #  W.to_excel('Item-Similarity-Norm.xlsx', 'Sheet1')

    return W


# Version1
def Recommend1(user, train, W, K, N):
    rank = dict()
    ru = train[user]
    for i, v in enumerate(ru[0]):  # i: index, v: item id, rate: ru[1][i]
        w = W.loc[v].sort_values(ascending=False)[:K]
        for j in w.index:
            if j in ru[0]:
                continue
            if j not in rank:
                rank[j] = 0
            rank[j] += w[j] * ru[1][i]
    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[0:N]  # list[(id, rating),()]


# Version2
def Recommend2(user, train, W, K, N):
    rank = dict()
    ru = train[user]
    for i, v in enumerate(ru[0]):  # i: index, v: item id, rate: ru[1][i]
        n = 0
        w = W[v].sort_values(ascending=False)
        for j in w.index:
            if j in ru[0]:
                continue
            if j not in rank:
                rank[j] = 0
            n += 1
            rank[j] += w[j] * ru[1][i]
            if n == K:
                break
    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[0:N]  # list[(id, rating),()]
