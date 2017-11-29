#!/usr/bin/env python
# -*- coding: utf-8 -*-

from multiprocessing import Process, Queue
import time
import evaluation as eva
import UserCF as UserCF
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


# UserCF ItemCF
def main():
    # 读取数据集
    file_path = '~/file/rs/dataset/ml-1m/ratings.dat'
    d_file = eva.readData(file_path, '::')
    M = 8  # 分组数
    N = 10  # 推荐个数
    K = [5, 10, 20, 40, 80, 120, 160]
    #  seeds = np.arange(M)
    seeds = [0]
    columns_list = [
        'PrecisionUserCF', 'PrecisionItemCF', 'RecallUserCF', 'RecallItemCF',
        'CoverageUserCF', 'CoverageItemCF', 'PopularityUserCF',
        'PopularityItemCF'
    ]
    userCF_columns = [
        'PrecisionUserCF', 'RecallUserCF', 'CoverageUserCF', 'PopularityUserCF'
    ]
    itemCF_columns = [
        'PrecisionItemCF', 'RecallItemCF', 'CoverageItemCF', 'PopularityItemCF'
    ]
    d = pd.DataFrame(
        np.zeros([len(K), len(columns_list)]), index=K, columns=columns_list)
    for index, seed in enumerate(seeds):
        train, test = eva.SplitData(d_file, M, seed)
        item_popularity = eva.ItemsPopularity(d_file, M, seed)
        W_user = UserCF.UserSimilarityVersion3(train)
        W_item = ItemCF.ItemSimilarityVersion2(train)
        for k in K:
            precision, recall, coverage, popularity = eva.PrecisionAndRecallAndCoverageAndPopularity(
                train, test, item_popularity, k, W_user, N, 1)
            d.loc[k, userCF_columns] += [
                precision, recall, coverage, popularity
            ]
            precision, recall, coverage, popularity = eva.PrecisionAndRecallAndCoverageAndPopularity(
                train, test, item_popularity, k, W_item, N, 2)
            d.loc[k, itemCF_columns] += [
                precision, recall, coverage, popularity
            ]
        d.loc[k] /= (index + 1)

    d.to_excel('Result-UserCF-ItemCF-K.xlsx', 'UserCF-ItemCF-K')
    fig, axes = plt.subplots(2, 2)
    axes[0][0].set_title('Precision')
    d.iloc[:, 0:2].plot(ax=axes[0][0], style=['o-', 'o-'])
    axes[0][1].set_title('Recall')
    d.iloc[:, 2:4].plot(ax=axes[0][1], style=['o-', 'o-'])
    axes[1][0].set_title('Coverage')
    d.iloc[:, 4:6].plot(ax=axes[1][0], style=['o-', 'o-'])
    axes[1][1].set_title('Popularity')
    d.iloc[:, 6:8].plot(ax=axes[1][1], style=['o-', 'o-'])

    plt.show()


# UserCF
def TestUserCF():
    # 读取数据集
    file_path = '~/file/rs/dataset/ml-1m/ratings.dat'
    d_file = eva.readData(file_path, '::')
    M = 8  # 分组数
    N = 10  # 推荐个数
    #  K = [5, 10]
    K = [5, 10, 20, 40, 80, 120, 160]
    #  seeds = np.arange(M)
    seeds = [0]
    columns_list = [
        'Precision', 'Recall', 'Coverage', 'Popularity'
    ]
    userCF_columns = [
        'Precision', 'Recall', 'Coverage', 'Popularity'
    ]
    d = pd.DataFrame(
        np.zeros([len(K), len(columns_list)]), index=K, columns=columns_list)
    for index, seed in enumerate(seeds):
        train, test = eva.SplitData(d_file, M, seed)
        item_popularity = eva.ItemsPopularity(d_file, M, seed)
        W_user = UserCF.UserSimilarityVersion3(train)
        for k in K:
            print(k)
            precision, recall, coverage, popularity = eva.PrecisionAndRecallAndCoverageAndPopularity(
                train, test, item_popularity, k, W_user, N, 1)
            d.loc[k, userCF_columns] += [
                precision, recall, coverage, popularity
            ]
        d.loc[k] /= (index + 1)

    d.to_excel('Result-UserCF-K.xlsx', 'UserCF-K')
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


# UserCF I and II
def TestUserCFIIF():
    file_path = '~/file/rs/dataset/ml-1m/ratings.dat'
    d_file = eva.readData(file_path, '::')
    M = 8  # 分组数
    N = 10  # 推荐个数
    #  K = [5, 10, 20]
    K = [[5, 40, 120], [10, 20, 80]]
    #  K = [5, 10, 20, 40, 80, 120, 160]
    #  seeds = np.arange(M)
    seeds = [0]
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
    for index, seed in enumerate(seeds):
        train, test = eva.SplitData(d_file, M, seed)
        item_popularity = eva.ItemsPopularity(d_file, M, seed)
        W_userI = UserCF.UserSimilarityVersion2(train)
        W_userII = UserCF.UserSimilarityVersion3(train)
        for k in K:
            precision, recall, coverage, popularity = eva.PrecisionAndRecallAndCoverageAndPopularity(
                train, test, item_popularity, k, W_userI, N, 1)
            d.loc[k, I_columns] += [
                precision, recall, coverage, popularity
            ]
            precision, recall, coverage, popularity = eva.PrecisionAndRecallAndCoverageAndPopularity(
                train, test, item_popularity, k, W_userII, N, 1)
            d.loc[k, II_columns] += [
                precision, recall, coverage, popularity
            ]
        d.loc[k] /= (index + 1)

    d.to_excel('Result-UserCF-I-II.xlsx', 'UserCF-I-II-K')
    fig, axes = plt.subplots(2, 2)
    axes[0][0].set_title('Precision')
    d.iloc[:, 0:2].plot(ax=axes[0][0], style=['o-', 'o-'])
    axes[0][1].set_title('Recall')
    d.iloc[:, 2:4].plot(ax=axes[0][1], style=['o-', 'o-'])
    axes[1][0].set_title('Coverage')
    d.iloc[:, 4:6].plot(ax=axes[1][0], style=['o-', 'o-'])
    axes[1][1].set_title('Popularity')
    d.iloc[:, 6:8].plot(ax=axes[1][1], style=['o-', 'o-'])

    plt.show()


def TestRandomMostPopupar():
    file_path = '~/file/rs/dataset/ml-1m/ratings.dat'
    d_file = eva.readData(file_path, '::')
    M = 8  # 分组数
    N = 10  # 推荐个数
    #  seeds = np.arange(M)
    seeds = [0]
    columns_list = [
        'Precision', 'Recall',
        'Coverage', 'Popularity'
    ]
    d = pd.DataFrame(
        np.zeros([2, len(columns_list)]), index=['Random', 'MostPopular'], columns=columns_list)
    for index, seed in enumerate(seeds):
        train, test = eva.SplitData(d_file, M, seed)
        item_popularity = eva.ItemsPopularity(d_file, M, seed)
        precision, recall, coverage, popularity = eva.RandomResult(
            train, test, item_popularity, N)
        d.loc['Random', columns_list] += [
            precision, recall, coverage, popularity
        ]
        d.loc['Random'] /= (index + 1)
        precision, recall, coverage, popularity = eva.MostPopularResult(
            train, test, item_popularity, N)
        d.loc['MostPopular', columns_list] += [
            precision, recall, coverage, popularity
        ]
        d.loc['MostPopular'] /= (index + 1)

    print(d)
    d.to_excel('Result-Random-Popular.xlsx', 'Random-Popular')


def WriteIntoD(q, d):
    print('Process to WriteIntoD')
    userCF_columns = [
        'Precision', 'Recall', 'Coverage', 'Popularity'
    ]
    n = 0
    while n != 6:
        value = q.get(True)
        print('Get %s from queue.' % value)
        d.loc[value[0], userCF_columns] += [
            value[1], value[2], value[3], value[4]
        ]
        n += 1
    d.to_excel('Result-UserCF-K.xlsx', 'UserCF-K')
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


def Evaluation(q, k, train, test, item_popularity, W, N):
    print('Process to k: %d' % k)
    start = time.time()
    precision, recall, coverage, popularity = eva.PrecisionAndRecallAndCoverageAndPopularity(
        train, test, item_popularity, k, W, N, 1)
    end = time.time()
    print('%d: %.2fs' % (k, end - start))
    q.put([k, precision, recall, coverage, popularity])


# UserCF I and II
def TestUserCFMult():
    start = time.time()
    file_path = '~/file/rs/dataset/ml-1m/ratings.dat'
    d_file = eva.readData(file_path, '::')
    M = 8  # 分组数
    N = 10  # 推荐个数
    K = [5, 10, 20, 40, 80, 120]
    #  K = [40, 80]
    #  seeds = np.arange(M)
    seeds = [0]
    columns_list = [
        'Precision', 'Recall', 'Coverage', 'Popularity'
    ]
    d = pd.DataFrame(
        np.zeros([len(K), len(columns_list)]), index=K, columns=columns_list)
    for index, seed in enumerate(seeds):
        train, test = eva.SplitData(d_file, M, seed)
        item_popularity = eva.ItemsPopularity(d_file, M, seed)
        W_user = UserCF.UserSimilarityVersion3(train)
        q = Queue()
        for k in K:
            pw = Process(target=Evaluation, args=(q, k, train, test, item_popularity, W_user, N))
            pw.start()  # 启动写
        pr = Process(target=WriteIntoD, args=(q, d))
        pr.start()  # 启动读
        pw.join()   # 等待pw结束
        end = time.time()
        print('Total Time: %.2fs' % (end - start))
        pr.join()  # 强制结束读


if __name__ == '__main__':
    if len(argv) != 2:
        print("Arg Error")
    elif argv[1] == 'user':
        TestUserCF()
    elif argv[1] == 'user12':
        TestUserCFIIF()
    elif argv[1] == 'random':
        TestRandomMostPopupar()
    elif argv[1] == 'main':
        main()
    elif argv[1] == 'mult':
        TestUserCFMult()
