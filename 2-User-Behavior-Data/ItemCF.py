#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd


def ItemSimilarity(train):
    C = dict()
    N = dict()

    for u, items in train.items():
        for i in items[0]:
            if i not in N:
                N[i] = 0
            if i not in C:
                C[i] = dict()
            N[i] += 1
            for j in items[0]:
                if i == j:
                    continue
                if j not in C[i]:
                    C[i][j] = 0
                C[i][j] += 1
    W = dict()
    for i, related_items in C.items():
        W[i] = dict()
        for j, cij in related_items.items():
            W[i][j] = cij / math.sqrt(N[i] * N[j])

    return pd.DataFrame(W)


def ItemSimilarityVersion2(train):
    C = dict()
    N = dict()

    for u, items in train.items():
        for i in items[0]:
            if i not in N:
                N[i] = 0
            if i not in C:
                C[i] = dict()
            N[i] += 1
            for j in items[0]:
                if i == j:
                    continue
                if j not in C[i]:
                    C[i][j] = 0
                C[i][j] += 1 / math.log(1 + len(items) * 1.0)
    W = dict()
    for i, related_items in C.items():
        W[i] = dict()
        for j, cij in related_items.items():
            W[i][j] = cij / math.sqrt(N[i] * N[j])

    return pd.DataFrame(W)


def Recommend(user, train, W, K, N):
    rank = dict()
    ru = train[user]
    for i, v in enumerate(ru[0]):  # i: index, v: item id, rate: ru[1][i]
        w = W[v].sort_values(ascending=False)[:K]
        for j in w.index:
            if j in ru[0]:
                continue
            if j not in rank:
                rank[j] = 0
            rank[j] += w[j] * ru[1][i]
    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[0:N]  # list[(id, rating),()]
