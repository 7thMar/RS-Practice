#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Create By K
# Long Tail

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as f
cnfont = f.FontProperties(
    fname='/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc')
matplotlib.rcParams['axes.unicode_minus'] = False

# plt.title('中文示例', fontproperties=cnfont)


def main():
    file_path = '~/file/rs/dataset/ml-1m/ratings.dat'
    #  file_path = '~/file/rs/dataset/ml-10m/ratings.dat'
    d_file = pd.read_csv(file_path, sep='::', usecols=[0, 1])

    # 用户活跃度 {userid: activity}
    activity = d_file.groupby('UserID').count()['MovieID'].to_dict()  # 各用户的电影数
    activity_count = pd.DataFrame(list(activity.values()), columns=['activity'])  # 电影数集
    activity_count['count'] = 1
    # 用户活跃度统计 {activity: count}
    activity_count = activity_count.groupby('activity').count()  # 统计各电影数数量

    # 物品流行度 {itemid: popularity}
    popularity = d_file.groupby('MovieID')['UserID'].count().to_dict()
    popularity_count = pd.DataFrame(list(popularity.values()), columns=['popularity'])
    popularity_count['count'] = 1
    # 物品流行度统计 {popularity: count}
    popularity_count = popularity_count.groupby('popularity').count()

    # 活跃度对应的流行度 {activity: popularity}
    activity_popularity = dict()
    d = d_file.groupby('UserID')['MovieID']
    for user, items in d:  # 遍历每个用户
        p = 0
        for item in items.tolist():  # 遍历每个用户的商品，求总物品流行度
            p += popularity[item]
        p /= len(items.tolist())  # 求物品流行度平均
        if activity[user] not in activity_popularity:
            activity_popularity[activity[user]] = 0
        activity_popularity[activity[user]] += p  # 将物品流行度加到该用户活跃度上

    activity_count_dict = activity_count['count'].to_dict()
    for k in activity_popularity:  # 平均
        activity_popularity[k] /= activity_count_dict[k]

    d1 = pd.Series(activity_popularity)

    fig, axes = plt.subplots(3, 1)
    popularity_count.plot(ax=axes[0], style='o')
    activity_count.plot(ax=axes[1], style='o')
    d1.plot(ax=axes[2], style='o')

    axes[0].set_title('物品流行度的长尾分布', fontproperties=cnfont)
    axes[0].set_xlabel('物品流行度', fontproperties=cnfont)
    axes[0].set_ylabel('物品数', fontproperties=cnfont)
    axes[1].set_title('用户活跃度的长尾分布', fontproperties=cnfont)
    axes[1].set_xlabel('用户活跃度', fontproperties=cnfont)
    axes[1].set_ylabel('用户数', fontproperties=cnfont)
    axes[2].set_title('用户活跃度和物品流行度的关系', fontproperties=cnfont)
    axes[2].set_xlabel('用户活跃度', fontproperties=cnfont)
    axes[2].set_ylabel('平均物品热门度', fontproperties=cnfont)
    plt.show()


if __name__ == '__main__':
    main()
