#!/usr/bin/env python
# -*- coding: utf-8 -*-

import evaluation as eva
import UserCF as UserCF
import ItemCF as ItemCF

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.font_manager as f
cnfont = f.FontProperties(
    fname='/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc')
matplotlib.rcParams['axes.unicode_minus'] = False

# plt.title('中文示例', fontproperties=cnfont)


def main():
    # 读取数据集
    file_path = '~/file/rs/dataset/ml-1m/ml-1m/ratings.dat'
    d_file = eva.readData(file_path, '::')
    M = 8  # 分组数
    N = 10  # 推荐个数
    #  K = [5, 10, 20]
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

    d.to_csv('UserCF&ItemCF.csv', sep=':')
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


if __name__ == '__main__':
    main()
