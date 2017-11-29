#!/usr/bin/env python
# -*- coding: utf-8 -*-

from multiprocessing import Process, Queue
from multiprocessing import Pool
import time
import ctypes as c
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
    F = 100  # 特征数
    N = 10  # 推荐数
    alpha = 0.02
    namda = 0.01
    ratio_arr = [20]
    #  ratio_arr = [20, 10, 5, 3, 2, 1]  # 负样本/正样本
    path = '~/file/rs/dataset/ml-1m/ratings.dat'  # 读取数据
    #  iter_num = 35
    iter_num = 35
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
        self.train_all_items = self.train_data.loc[:, 'MovieID'].tolist()
        train_group = self.train_data.groupby("UserID")["MovieID"]
        train = dict()  # 各用户 DataFrame
        for k, v in train_group:
            train[k] = v.loc[:].tolist()
        self.train_user_list = list(train.keys())

        test_data = data.iloc[test_index, :]
        test_group = test_data.groupby("UserID")["MovieID"]
        test = dict()  # 'userID, DataFrame'
        for k, v in test_group:
            test[k] = v.loc[:].tolist()
        self.test_user_list = list(test.keys())
        self.popularity = self.train_data.groupby("MovieID")["UserID"].count().to_dict()
        return train, test

    # P189
    # 将两矩阵随机数填充
    # 和 sqrt(F) 成正比
    def InitModel(self):
        self.P = np.random.rand(len(self.train_user_list), self.F) / math.sqrt(self.F)
        self.Q = np.random.rand(len(self.train_item_list), self.F) / math.sqrt(self.F)

    def RandomSelectNegativeSample_AllUser(self, ratio, train):
        samples_positive = [None] * len(self.train_user_list)
        samples_negative = [None] * len(self.train_user_list)
        items_pool = self.train_all_items  # 没有将user已经有的删除，缩短运行时间
        candidate_length = len(items_pool)
        c_items_pool = (c.c_int * candidate_length)(*items_pool)
        for i, v in enumerate(self.train_user_list):
            print('ratio: ', ratio, 'user: ', i)
            samples_positive[i] = train[v]
            negative_sample_length = ratio * len(samples_positive[i])
            items_index = np.random.randint(candidate_length, size=negative_sample_length * 3)
            index_length = 3 * negative_sample_length
            c_items_index = (c.c_int * index_length)(*items_index)
            c_positive = (c.c_int * len(samples_positive))(*samples_positive[i])
            c_negative = (c.c_int * negative_sample_length)()
            n = self.lfm.SelectNegativeSample(c_items_pool, c_items_index, c_positive, c_negative, negative_sample_length)
            samples_negative[i] = np.fromiter(c_negative, dtype=np.int)[:n]
        return samples_positive, samples_negative

    def C_LatentFactorModel(self, alpha, namda, ratio, train):
        self.lfm = c.cdll.LoadLibrary('./LFM.so')
        self.lfm.UpdatePQPositive.argtypes = c.POINTER(c.c_float), c.POINTER(c.c_float), c.c_int, c.c_float, c.c_float
        self.lfm.UpdatePQPositive.restype = None
        self.lfm.UpdatePQNegative.argtypes = c.POINTER(c.c_float), c.POINTER(c.c_float), c.c_int, c.c_float, c.c_float
        self.lfm.UpdatePQNegative.restype = None
        c.c_float, c.c_float
        self.lfm.UpdatePQAll.restype = None
        self.lfm.SelectNegativeSample.argtypes = c.POINTER(c.c_int), c.POINTER(c.c_int), c.POINTER(c.c_int), c.POINTER(c.c_int), c.c_int
        self.lfm.SelectNegativeSample.restype = c.c_int
        self.InitModel()  # Matrix user*F item*F
        #  iter_num * train.lenth * ratio * items[user].lenth
        for step in range(0, self.iter_num):
            samples_positive, samples_negative = self.RandomSelectNegativeSample_AllUser(ratio, train)
            for i, user in enumerate(self.train_user_list):
                p_c = (c.c_float * self.F)(*self.P[i])
                for item in samples_positive[i]:
                    item_index = self.train_item_list.index(item)
                    q_c = (c.c_float * self.F)(*self.Q[item_index])
                    self.lfm.UpdatePQPositive(p_c, q_c, self.F, alpha, namda)
                    self.Q[item_index] = np.fromiter(q_c, dtype=np.float)
                for item in samples_negative[i]:
                    item_index = self.train_item_list.index(item)
                    q_c = (c.c_float * self.F)(*self.Q[item_index])
                    self.lfm.UpdatePQNegative(p_c, q_c, self.F, alpha, namda)
                    self.Q[item_index] = np.fromiter(q_c, dtype=np.float)
                self.P[i] = np.fromiter(p_c, dtype=np.float)
                print('ratio: ', ratio, 'step:', step, ' user: ', user)
            alpha *= 0.9

    def PrecisionAndRecallAndCoverageAndPopularity(self, train, test, user_item, item_popularity, N):
        hit = 0  # 推荐正确数
        rank_num = 0  # 推荐总数
        test_items_num = 0  # 测试集总商品数
        recommend_items = set()
        popularity = 0.0
        for user in self.train_user_list:  # test
            if user not in self.test_user_list:
                continue
            having_list = train[user]
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
                len(self.train_item_list) * 1.0), popularity / (rank_num * 1.0)

    def Recommend(self, user, user_item):
        return user_item.sort_values(user, ascending=False).index

    def Evaluation(self, ratio, train, test, item_popularity, N):
        print('Process to ratio: %d' % ratio)
        self.C_LatentFactorModel(self.alpha, self.namda, ratio, train)
        self.P = pd.DataFrame(self.P, index=self.train_user_list)
        self.Q = pd.DataFrame(self.Q, index=self.train_item_list)
        user_item = self.P.dot(self.Q.T).T  # columns: userid, index: itemid
        precision, recall, coverage, popularity = self.PrecisionAndRecallAndCoverageAndPopularity(
            train, test, user_item, item_popularity, N)
        return [precision, recall, coverage, popularity]

    def Result(self):
        train, test = self.ReadAndSplitData()
        columns = [
            'Precision', 'Recall', 'Coverage', 'Popularity'
        ]
        d = pd.DataFrame(
            np.zeros([len(self.ratio_arr), len(columns)]), index=self.ratio_arr, columns=columns)
        p = Pool(4)
        result = dict()
        for ratio in self.ratio_arr:
            result[ratio] = p.apply_async(self.Evaluation, args=(ratio, train, test, self.popularity, self.N))
        p.close()  # 启动读
        p.join()   # 等待p结束
        for k, v in result.items():
            d.loc[k, columns] += v.get()
        d.to_excel('Result-LFM-K.xlsx', 'LFM-K')
        fig, axes = plt.subplots(2, 2)
        axes[0][0].set_title('LFM 准确度与 ratio 的关系', fontproperties=cnfont)
        axes[0][1].set_title('LFM 召回率与 ratio 的关系', fontproperties=cnfont)
        axes[1][0].set_title('LFM 覆盖率与 ratio 的关系', fontproperties=cnfont)
        axes[1][1].set_title('LFM 流行度与 ratio 的关系', fontproperties=cnfont)
        d.iloc[:, :1].plot(ax=axes[0][0], style=['o-'], xticks=[0, 1, 2, 3, 5, 10, 20, 25])
        d.iloc[:, 1:2].plot(ax=axes[0][1], style=['o-'], xticks=[0, 1, 2, 3, 5, 10, 20, 25])
        d.iloc[:, 2:3].plot(ax=axes[1][0], style=['o-'], xticks=[0, 1, 2, 3, 5, 10, 20, 25])
        d.iloc[:, 3:].plot(ax=axes[1][1], style=['o-'], xticks=[0, 1, 2, 3, 5, 10, 20, 25])
        plt.show()


def main():
    lfm = LFM()
    lfm.Result()


if __name__ == '__main__':
    main()
