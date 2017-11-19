#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ctypes as c
import evaluation as eva
import math
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as f
cnfont = f.FontProperties(
    fname='/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc')
matplotlib.rcParams['axes.unicode_minus'] = False

# plt.title('中文示例', fontproperties=cnfont)


class LFM(object):
    F = 100
    N = 5
    alpha = 0.02
    namda = 0.01
    ratio_arr = [1, 2, 3, 5, 10, 20]  # 负样本/正样本
    path = '~/file/rs/dataset/ml-1m/ratingsTest.dat'  # 读取数据
    #  iter_num = [5, 10, 15, 20]
    iter_num = 10
    sample_i = 1
    F_Index = np.arange(100)

    # train/test user_list
    # train/test item_list
    # P Q
    # train
    # test

    def __init__(self):
        pass

    # train: [userid] = list[items]
    # test:  [userid] = list[items]
    def ReadAndSplitData(self):
        data = pd.read_csv(self.path, sep='::', usecols=[0, 1])
        count = data.shape[0]  # 数据总数
        np.random.seed(0)  # seed
        index = np.arange(count)
        test_index = np.random.choice(index, int(count / 8), replace=False)
        train_index = np.setdiff1d(index, test_index)

        self.train_data = data.iloc[train_index, :]
        self.train_item_list = self.train_data['MovieID'].value_counts().index.tolist()
        self.train_all_item = self.train_data.loc[:, 'MovieID'].tolist()
        train_group = self.train_data.groupby("UserID")["MovieID"]
        self.train = dict()  # 各用户 DataFrame
        for k, v in train_group:
            self.train[k] = v.loc[:].tolist()
        self.train_user_list = self.train.keys()

        test_data = data.iloc[test_index, :]
        test_group = test_data.groupby("UserID")["MovieID"]
        self.test = dict()  # 'userID, DataFrame'
        for k, v in test_group:
            self.test[k] = v.loc[:].tolist()
        self.test_user_list = self.test.keys()
        self.popularity = self.train_data.groupby("MovieID")["UserID"].count().to_dict()

    # P189
    # 将两矩阵随机数填充
    # 和 sqrt(F) 成正比
    def InitModel(self):
        self.P = np.random.rand(len(self.train_user_list), self.F) / math.sqrt(self.F)
        self.Q = np.random.rand(len(self.train_item_list), self.F) / math.sqrt(self.F)

    def RandomSelectNegativeSample_AllUser(self, ratio):
        samples_positive = [None] * len(self.train_user_list)
        samples_negative = [None] * len(self.train_user_list)
        for i, v in enumerate(self.train_user_list):
            print(i)
            samples_positive[i] = self.train[v]
            items_pool = self.train_all_item
            candidate_length = len(items_pool)
            negative_sample_length = ratio * len(samples_positive[i])
            negative = np.arange(0, 1).repeat(negative_sample_length)
            n = 0
            for _ in range(0, negative_sample_length * 3):
                item = items_pool[np.random.randint(0, candidate_length)]
                if item in negative or item in samples_positive[i]:
                    continue
                negative[n] = item
                n += 1
                if n >= negative_sample_length:
                    break
            samples_negative[i] = negative[:n]
        self.sample_i += 1
        return samples_positive, samples_negative

    def C_LatentFactorModel(self, alpha, namda, ratio):
        lfm = c.cdll.LoadLibrary('./LFM.so')
        lfm.UpdatePQPositive.argtypes = c.POINTER(c.c_float), c.POINTER(c.c_float), c.c_int, c.c_float, c.c_float
        lfm.UpdatePQPositive.restype = None
        lfm.UpdatePQNegative.argtypes = c.POINTER(c.c_float), c.POINTER(c.c_float), c.c_int, c.c_float, c.c_float
        lfm.UpdatePQNegative.restype = None
        lfm.UpdatePQAll.argtypes = c.POINTER(c.POINTER(c.c_int)), c.POINTER(c.POINTER(c.c_int)), c.POINTER(c.POINTER(c.c_float)), c.POINTER(c.POINTER(c.c_float)), c.c_int, c.POINTER(c.c_int), c.c_int,
        c.c_float, c.c_float
        lfm.UpdatePQAll.restype = None
        self.InitModel()  # DataFrame user*F item*F
        #  iter_num * train.lenth * ratio * items[user].lenth
        for step in range(0, self.iter_num):
            samples_positive, samples_negative = self.RandomSelectNegativeSample_AllUser(ratio)
            for i, user in enumerate(self.train_user_list):
                p_c = (c.c_float * self.F)(*self.P[i])
                for item in samples_positive[i]:
                    item_index = self.train_item_list.index(item)
                    q_c = (c.c_float * self.F)(*self.Q[item_index])
                    lfm.UpdatePQPositive(p_c, q_c, self.F, alpha, namda)
                    self.Q[item_index] = np.fromiter(q_c, dtype=np.float)
                for item in samples_negative[i]:
                    item_index = self.train_item_list.index(item)
                    q_c = (c.c_float * self.F)(*self.Q[item_index])
                    lfm.UpdatePQNegative(p_c, q_c, self.F, alpha, namda)
                    self.Q[item_index] = np.fromiter(q_c, dtype=np.float)
                self.P[i] = np.fromiter(p_c, dtype=np.float)
                print('step:', step, ' user: ', user)
            alpha *= 0.9

    def PrecisionAndRecallAndCoverageAndPopularity(self, train, test, user_item, item_popularity, N):
        hit = 0  # 推荐正确数
        rank_num = 0  # 推荐总数
        test_items_num = 0  # 测试集总商品数
        recommend_items = set()
        all_items = set()
        popularity = 0.0
        for user in self.train_user_list:  # test
            if user not in self.test_user_list:
                continue
            having_list = train[user]
            all_items = all_items | set(having_list)
            test_items = test[user]
            rank = self.Recommend(user, user_item)  # 所有排序商品
            n = 0  # 推荐N个
            for item in rank:
                if item in having_list:
                    continue
                if item in test_items:
                    hit += 1
                popularity += math.log(1 + item_popularity[item])
                recommend_items.add(item)
                n += 1
                if n == N:
                    break
            rank_num += n
            test_items_num += len(test_items)
        return hit / (rank_num * 1.0), hit / (
            test_items_num * 1.0), len(recommend_items) / (
                len(all_items) * 1.0), popularity / (rank_num * 1.0)

    def Recommend(self, user, user_item):
        return user_item.sort_values(user, ascending=False).index

    def Result(self):
        self.ReadAndSplitData()
        columns_list = ['Precision', 'Recall', 'Coverage', 'Popularity']
        d = pd.DataFrame(np.zeros([len(self.ratio_arr), len(columns_list)]), index=self.ratio_arr, columns=columns_list)
        for ratio in self.ratio_arr:
            self.C_LatentFactorModel(self.alpha, self.namda, ratio)
            self.P = pd.DataFrame(self.P, index=self.train_user_list)
            self.Q = pd.DataFrame(self.Q, index=self.train_item_list)
            user_item = self.P.dot(self.Q.T).T  # columns: userid, index: itemid
            user_item.to_csv('T.csv', sep='|')
            precision, recall, coverage, popularity = self.PrecisionAndRecallAndCoverageAndPopularity(
                self.train, self.test, user_item, self.popularity, self.N)
            d.loc[ratio] = [precision, recall, coverage, popularity]
            #  print('Precision: ', precision)
            #  print('Recall: ', recall)
            #  print('Coverage: ', coverage)
            #  print('Popularity: ', popularity)

        fig, axes = plt.subplots(1, 1)
        axes[0].set_title('LFM 各指标与 ratio 的关系', fontproperties=cnfont)
        d.iloc[:, :].plot(ax=axes[0], style=['o-', 'o-', 'o-', 'o-'])
        axes[0].set_xlabel('ratio')
        plt.show()

def main():
    lfm = LFM()
    lfm.Result()


if __name__ == '__main__':
    main()
