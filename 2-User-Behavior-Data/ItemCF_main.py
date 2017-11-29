#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Create By K
from multiprocessing import Process, Queue, Pool
import math
import time
import evaluation as eva
import ItemCF as ItemCF

from sys import argv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.font_manager as f
cnfont = f.FontProperties(
    fname='/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc')
matplotlib.rcParams['axes.unicode_minus'] = False

# plt.title('中文示例', fontproperties=cnfont)


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
    train_items_list = data_train['MovieID'].value_counts().index.tolist()
    d_items_user = data_train.groupby("MovieID")["UserID"]
    item_popularity = d_items_user.count().to_dict()
    #  item_popularity = data_train.groupby("MovieID")["UserID"].count().to_dict()

    items_user = dict()
    for k, v in d_items_user:
        items_user[k] = set(v.tolist())

    for k, v in train_group:
        train[k] = v.as_matrix(["MovieID", "Rating"]).T

    data_test = data.iloc[test_index, :]
    test_group = data_test.groupby("UserID")["MovieID", "Rating"]
    for k, v in test_group:
        test[k] = v.as_matrix(["MovieID", "Rating"]).T

    return train, test, train_items_list, item_popularity, items_user


# N: 物品数目
# train: {user: item, }
# 和推荐数目N有很大的关系
# 准确率 召回率 覆盖率 流行度
def PrecisionAndRecallAndCoverageAndPopularity(train, test, item_popularity, K, W, N, method):
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
        tu = test[user][0]
        if method == 1:
            rank = ItemCF.Recommend1(user, train, W, K, N)
        elif method == 2:
            rank = ItemCF.Recommend2(user, train, W, K, N)
        for item, _ in rank:
            if item in tu:
                hit += 1
            popularity += math.log(1 + item_popularity[item])
            recommend_items.add(item)
        num_rank += len(rank)
        num_tu += len(tu)
    print(K, ': ', hit / (num_rank * 1.0), hit / (num_tu * 1.0), len(recommend_items) / (len(all_items) * 1.0), popularity / (num_rank * 1.0))

    return hit / (num_rank * 1.0), hit / (
        num_tu * 1.0), len(recommend_items) / (len(all_items) * 1.0), popularity / (num_rank * 1.0)


def Evaluation(k, train, test, item_popularity, W, N, method=1):
    print('Process to k: %d' % k)
    start = time.time()
    precision, recall, coverage, popularity = PrecisionAndRecallAndCoverageAndPopularity(
        train, test, item_popularity, k, W, N, method)
    end = time.time()
    print('%d: %.2fs' % (k, end - start))
    return [precision, recall, coverage, popularity]


# [588, 2924, 1, 2762, 2571, 356]
def PrintMovieSimilarity():
    user_path = '~/file/rs/dataset/ml-1m/ratings.dat'
    movie_path = '~/file/rs/dataset/ml-1m/movies.dat'
    d_file = pd.read_csv(user_path, sep='::', usecols=[0, 1])
    d_movie = pd.read_csv(movie_path, sep='::', index_col=0, usecols=[0, 1])
    train_group = d_file.groupby("UserID")["MovieID"]
    train = dict()
    for k, v in train_group:
        train[k] = v.tolist()

    C = dict()
    N = dict()

    for u, items in train.items():
        for i in items:
            if i not in N:
                N[i] = 0
            if i not in C:
                C[i] = dict()
            N[i] += 1
            for j in items:
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

    w_item = pd.DataFrame(W)
    movies = [588, 2924, 1, 2762, 2571, 356]
    d = pd.DataFrame(columns=['Movie-1', 'Movie-2', 'Similarity'])
    n = 0
    for i in movies:
        l = w_item.loc[i].sort_values(ascending=False).index[:5]
        for j in l:
            d.loc[n] = [d_movie.at[i, 'Title'], d_movie.at[j, 'Title'], w_item.at[i, j]]
            n += 1
    d.to_excel('Result-Movie-Similarity.xlsx', 'Sheet1')


def TestItemCF():
    start = time.time()
    path = '~/file/rs/dataset/ml-1m/ratingsTest.dat'
    d_file = pd.read_csv(path, sep='::', usecols=[0, 1, 2])
    M = 8  # 分组数
    N = 10  # 推荐个数
    K = [5]
    #  K = [5, 10, 20, 40]
    #  K = [5, 10, 20, 40, 80, 120, 160]
    train, test, train_items_list, item_popularity, items_user = SplitData(d_file, M, 1)  # 0: seed
    #  w_item = ItemCF.ItemSimilarityVersion1(train, item_popularity, items_user)
    w_item = ItemCF.ItemSimilarityVersion2(train, item_popularity, items_user)
    columns_list = [
        'precision', 'recall', 'coverage', 'popularity'
    ]
    d = pd.DataFrame(
        np.zeros([len(K), len(columns_list)]), index=K, columns=columns_list)
    p = Pool(4)
    result = dict()
    for k in K:
        result[k] = p.apply_async(Evaluation, args=(k, train, test, item_popularity, w_item, N, 2))
    p.close()
    p.join()  # 等待所有子进程执行完毕
    for k, v in result.items():
        d.loc[k, columns_list] += v.get()
    end = time.time()
    print('total time: %.2fs' % (end - start))

    d.to_excel('Test.xlsx', 'ItemCF-K')
    #  d.to_excel('Result-ItemCF-K.xlsx', 'ItemCF-K')
    fig, axes = plt.subplots(2, 2)
    axes[0][0].set_title('Precision')
    axes[0][0].plot(d.iloc[:, 0], 'o-', label='precision')
    axes[0][1].set_title('Recall')
    axes[0][1].plot(d.iloc[:, 1], 'o-', label='recall')
    axes[1][0].set_title('Coverage')
    axes[1][0].plot(d.iloc[:, 2], 'o-', label='coverage')
    axes[1][1].set_title('Popularity')
    axes[1][1].plot(d.iloc[:, 3], 'o-', label='popularity')
    plt.legend()
    plt.show()


def TestItemCF_IUF():
    file_path = '~/file/rs/dataset/ml-1m/ratings.dat'
    d_file = eva.readData(file_path, '::')
    M = 8  # 分组数
    N = 10  # 推荐个数
    K = [5, 10, 20, 40, 80, 120, 160]
    #  K = [5, 10, 20, 40]
    train, test, train_items_list, item_popularity, items_user = SplitData(d_file, M, 0)  # 0: seed
    p = Pool(4)
    W_ItemCF = p.apply_async(ItemCF.ItemSimilarityVersion1, args=(train, item_popularity, items_user)).get()
    W_IUF = p.apply_async(ItemCF.ItemSimilarityVersion2, args=(train, item_popularity, items_user)).get()
    #  W_ItemCF = ItemCF.ItemSimilarityVersion1(train)
    #  W_IUF = ItemCF.ItemSimilarityVersion2(train)
    p.close()
    p.join()
    columns_list = [
        'Precision-ItemCF', 'Precision-IUF', 'Recall-ItemCF', 'Recall-IUF',
        'Coverage-ItemCF', 'Coverage-IUF', 'Popularity-ItemCF',
        'Popularity-IUF'
    ]
    I_columns = [
        'Precision-ItemCF', 'Recall-ItemCF', 'Coverage-ItemCF', 'Popularity-ItemCF'
    ]
    II_columns = [
        'Precision-IUF', 'Recall-IUF', 'Coverage-IUF', 'Popularity-IUF'
    ]
    d = pd.DataFrame(
        np.zeros([len(K), len(columns_list)]), index=K, columns=columns_list)

    # ItemCF
    p = Pool(4)
    resultItemCF = dict()
    resultItemIUF = dict()
    for k in K:
        resultItemCF[k] = p.apply_async(Evaluation, args=(k, train, test, item_popularity, W_ItemCF, N))
        resultItemIUF[k] = p.apply_async(Evaluation, args=(k, train, test, item_popularity, W_IUF, N))
    p.close()
    p.join()  # 等待所有子进程执行完毕
    for k, v in resultItemCF.items():
        d.loc[k, I_columns] += v.get()
    for k, v in resultItemIUF.items():
        d.loc[k, II_columns] += v.get()

    d.to_excel('Result-ItemCF-IUF-K.xlsx', 'ItemCF-K')
    fig, axes = plt.subplots(2, 2)
    axes[0][0].set_title('Precision')
    d.iloc[:, 0:2].plot(ax=axes[0][0], style=['o-', 'o-'])
    axes[0][1].set_title('Recall')
    d.iloc[:, 2:4].plot(ax=axes[0][1], style=['o-', 'o-'])
    axes[1][0].set_title('Coverage')
    d.iloc[:, 4:6].plot(ax=axes[1][0], style=['o-', 'o-'])
    axes[1][1].set_title('Popularity')
    d.iloc[:, 6:8].plot(ax=axes[1][1], style=['o-', 'o-'])
    plt.legend()
    plt.show()


def TestItemCF_Norm():
    file_path = '~/file/rs/dataset/ml-1m/ratings.dat'
    start = time.time()
    d_file = eva.readData(file_path, '::')
    M = 8  # 分组数
    N = 10  # 推荐个数
    #  K = [10]
    K = [5, 10, 20, 40, 80, 120, 160]
    train, test, train_items_list, item_popularity, items_user = SplitData(d_file, M, 0)  # 0: seed
    W_ItemCF = ItemCF.ItemSimilarityVersion1(train, item_popularity, items_user)
    W_Norm = ItemCF.ItemSimilarityNorm(train, item_popularity, items_user)
    columns_list = [
        'Precision-ItemCF', 'Precision-Norm', 'Recall-ItemCF', 'Recall-Norm',
        'Coverage-ItemCF', 'Coverage-Norm', 'Popularity-ItemCF',
        'Popularity-Norm'
    ]
    I_columns = [
        'Precision-ItemCF', 'Recall-ItemCF', 'Coverage-ItemCF', 'Popularity-ItemCF'
    ]
    II_columns = [
        'Precision-Norm', 'Recall-Norm', 'Coverage-Norm', 'Popularity-Norm'
    ]
    d = pd.DataFrame(
        np.zeros([len(K), len(columns_list)]), index=K, columns=columns_list)

    # ItemCF
    p = Pool(4)
    resultItemCF = dict()
    resultNorm = dict()
    for k in K:
        resultItemCF[k] = p.apply_async(Evaluation, args=(k, train, test, item_popularity, W_ItemCF, N))
        resultNorm[k] = p.apply_async(Evaluation, args=(k, train, test, item_popularity, W_Norm, N))
    p.close()
    p.join()  # 等待所有子进程执行完毕
    for k, v in resultItemCF.items():
        d.loc[k, I_columns] += v.get()
    for k, v in resultNorm.items():
        d.loc[k, II_columns] += v.get()

    end = time.time()
    print('total time: %.2fs' % (end - start))

    d.to_excel('Result-ItemCF-Norm-K.xlsx', 'ItemCF-K')
    fig, axes = plt.subplots(2, 2)
    axes[0][0].set_title('Precision')
    d.iloc[:, 0:2].plot(ax=axes[0][0], style=['o-', 'o-'])
    axes[0][1].set_title('Recall')
    d.iloc[:, 2:4].plot(ax=axes[0][1], style=['o-', 'o-'])
    axes[1][0].set_title('Coverage')
    d.iloc[:, 4:6].plot(ax=axes[1][0], style=['o-', 'o-'])
    axes[1][1].set_title('Popularity')
    d.iloc[:, 6:8].plot(ax=axes[1][1], style=['o-', 'o-'])
    plt.legend()
    plt.show()


# 测试原版推荐算法和新版推荐算法
def test_recommend():
    file_path = '~/file/rs/dataset/ml-1m/ratings.dat'
    d_file = eva.readData(file_path, '::')
    M = 8  # 分组数
    N = 10  # 推荐个数
    K = [5, 10, 20, 40, 80, 120, 160]
    #  K = [5, 10, 20, 40]
    train, test, train_items_list, item_popularity, items_user = SplitData(d_file, M, 0)  # 0: seed
    #  W = ItemCF.ItemSimilarityVersion1(train, item_popularity, items_user)
    W = ItemCF.ItemSimilarityVersion2(train, item_popularity, items_user)
    columns_list = [
        'Precision-I', 'Precision-II', 'Recall-I', 'Recall-II',
        'Coverage-I', 'Coverage-II', 'Popularity-I',
        'Popularity-II'
    ]
    I_columns = [
        'Precision-I', 'Recall-I', 'Coverage-I', 'Popularity-I'
    ]
    II_columns = [
        'Precision-II', 'Recall-II', 'Coverage-II', 'Popularity-II'
    ]
    d = pd.DataFrame(
        np.zeros([len(K), len(columns_list)]), index=K, columns=columns_list)

    # ItemCF
    p = Pool(4)
    resultI = dict()
    resultII = dict()
    for k in K:
        resultI[k] = p.apply_async(Evaluation, args=(k, train, test, item_popularity, W, N, 1))
        resultII[k] = p.apply_async(Evaluation, args=(k, train, test, item_popularity, W, N, 2))
    p.close()
    p.join()  # 等待所有子进程执行完毕
    for k, v in resultI.items():
        d.loc[k, I_columns] += v.get()
    for k, v in resultII.items():
        d.loc[k, II_columns] += v.get()

    d.to_excel('Result-ItemCF-Recommend-K.xlsx', 'ItemCF-K')
    fig, axes = plt.subplots(2, 2)
    axes[0][0].set_title('Precision')
    d.iloc[:, 0:2].plot(ax=axes[0][0], style=['o-', 'o-'])
    axes[0][1].set_title('Recall')
    d.iloc[:, 2:4].plot(ax=axes[0][1], style=['o-', 'o-'])
    axes[1][0].set_title('Coverage')
    d.iloc[:, 4:6].plot(ax=axes[1][0], style=['o-', 'o-'])
    axes[1][1].set_title('Popularity')
    d.iloc[:, 6:8].plot(ax=axes[1][1], style=['o-', 'o-'])
    plt.legend()
    plt.show()


def main():
    pass


if __name__ == '__main__':
    if len(argv) != 2:
        print("Arg Error")
    elif argv[1] == '1':
        TestItemCF()
    elif argv[1] == '12':
        TestItemCF_IUF()
    elif argv[1] == 'norm':
        TestItemCF_Norm()
    elif argv[1] == 'movie':
        PrintMovieSimilarity()
    elif argv[1] == 're':
        test_recommend()
