#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math


# 评分预测
def RMSE(records):
    return math.sqrt(sum([(rui - pui) * (rui - pui) for u, i, rui, pui in records]) / float(len(records)))


def MAE(records):
    return sum([abs(rui - pui) for u, i, rui, pui in records]) / float(len(records))


# TopN 推荐
# 召回率和准确率
def precision_recall(test, N):
    hit = 0
    n_recall = 0
    n_precision = 0
    for user, item in test.items():
        rank = Recommand(user, N)
        hit += len(rank & items)
        n_recall += len(items)
        n_precision += len(rank)
    # 求出总的召回率和准确率
    return [hit / (1.0 * n_recall), hit / (1.0 * n_precision)]
