#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import time
from functools import wraps
import numpy as np
import pandas as pd


def fn_timer(func):

    @wraps(func)
    def timer(*args, **kwargs):
        t_start = time.time()
        result = func(*args, **kwargs)
        t_close = time.time()
        print("Total time running %s: %s seconds" % (func.__name__,
                                                     str(t_close - t_start)))
        return result

    return timer


#  @fn_timer
def UserSimilarityVersion1(train):
    W = dict()
    for u in train:
        W[u] = dict()
        for v in train:
            if u == v:
                continue
            length = len(np.intersect1d(train[u][0], train[v][0]))
            W[u][v] = (length * 1.0) / math.sqrt(
                len(train[u][0]) * len(train[v][0]) * 1.0)
    return W


def UserSimilarityVersion2(train):
    item_users = dict()
    #  商品: 用户
    for u, items in train.items():
        for i in items[0]:
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)

    C = dict()
    N = dict()
    for i, users in item_users.items():
        for u in users:
            if u not in N:
                N[u] = 0
            if u not in C:
                C[u] = dict()
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                if v not in C[u]:
                    C[u][v] = 0
                C[u][v] += 1

    # 计算相似度
    W = dict()
    for u, related_users in C.items():
        W[u] = dict()
        for v, cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])

    return pd.DataFrame(W)


def UserSimilarityVersion3(train):
    item_users = dict()
    #  商品: 用户
    for u, items in train.items():
        for i in items[0]:
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)

    C = dict()
    N = dict()
    for i, users in item_users.items():
        for u in users:
            if u not in N:
                N[u] = 0
            if u not in C:
                C[u] = dict()
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                if v not in C[u]:
                    C[u][v] = 0
                C[u][v] += 1 / math.log(1 + len(users))

    # 计算相似度
    W = dict()
    for u, related_users in C.items():
        W[u] = dict()
        for v, cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])

    return pd.DataFrame(W)


# K: 兴趣最接近的K个用户
def Recommend(user, train, W, K, N):
    rank = dict()
    interacted_items = train[user][0]  # 拥有的商品

    w = W[user].sort_values(ascending=False)[:K]
    d_arr = []
    for v in w.index:
        available_items = np.setdiff1d(train[v][0], interacted_items)
        d = pd.DataFrame(available_items, columns=["item"])
        d["w"] = w[v]
        d_arr.append(d)
    d_total = pd.concat(d_arr)
    d_groupby = d_total.groupby("item")["w"].sum()
    rank = d_groupby.sort_values(ascending=False)[:N].index
    return rank
    #  rank = d_groupby.sort_values(ascending=False)[:N]
    #  return rank.to_dict()

    #  for v, wuv in sorted(
            #  W[user].items(), key=lambda d: d[1], reverse=True)[0:K]:
        #  for i, rvi in zip(train[v][0], train[v][1]):
            #  if i in interacted_items:
                #  continue
            #  if i not in rank:
                #  rank[i] = 0
            #  rank[i] += wuv * rvi
    #  return sorted(rank.items(), key=lambda d: d[1], reverse=True)[0:N]
