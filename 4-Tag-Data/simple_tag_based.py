#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Create By K

import time
import math
import collections
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.font_manager as f
cnfont = f.FontProperties(fname='/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc')
matplotlib.rcParams['axes.unicode_minus'] = False
# plt.title('中文示例', fontproperties=cnfont)


class Tag(object):
    M = 10

    def __init__(self):
        pass

    def read_data(self, path, sep):
        return pd.read_csv(path, sep=sep, usecols=[0, 1, 2])

    def split_data(self, d, M):
        start = time.time()
        count = d.shape[0]  # 数据总数
        all_index = np.arange(count)
        group = d.groupby(['user', 'bookmark'])
        n = 0
        np.random.seed(0)
        index = np.random.randint(10, size=len(group))
        train_index = []
        test_index = []
        for k, v in group:
            if index[n] == 0:
                test_index += v.index.tolist()
            n += 1
        train_index = list(set(all_index) - set(test_index))
        d_train = d.iloc[train_index, :]
        #  user_list = d_train['user'].value_counts().index.tolist()
        #  tag_list = d_train['tag'].value_counts().index.tolist()
        #  item_list = d_train['bookmark'].value_counts().index.tolist()

        user_tags_group = d_train.groupby(['user', 'tag'])['bookmark'].count()
        user_tags_index = user_tags_group.index.tolist()
        user_tags_value = user_tags_group.values.tolist()
        user_tags = dict()
        for i, v in enumerate(user_tags_index):
            if v[0] not in user_tags:
                user_tags[v[0]] = dict()
            user_tags[v[0]][v[1]] = user_tags_value[i]

        item_tags_group = d_train.groupby(['bookmark', 'tag'])['user'].count()
        item_tags_index = item_tags_group.index.tolist()
        item_tags_value = item_tags_group.values.tolist()
        item_tags = dict()
        for i, v in enumerate(item_tags_index):
            if v[0] not in item_tags:
                item_tags[v[0]] = dict()
            item_tags[v[0]][v[1]] = item_tags_value[i]

        user_items_group = d_train.groupby('user')['bookmark']
        user_items = dict()
        for k, v in user_items_group:
            user_items[k] = v.tolist()
        train = user_items

        d_test = d.iloc[test_index, :]
        test = dict()
        test_group = d_test.groupby('user')['bookmark']
        for k, v in test_group:
            test[k] = v.tolist()

        item_pop = d_train.groupby('bookmark')['user'].count().to_dict()

        end = time.time()
        print("Split data %.2fs" % (end - start))
        return train, test, user_items, user_tags, item_tags, item_pop

    # N: 物品数目
    # train: {user: item, }
    # 准确率 召回率 覆盖率  多样性 流行度
    def evaluation(self, train, test, N):
        hit = 0
        num_rank = 0
        num_tu = 0
        recommend_items = set()
        all_items = set()
        popularity = 0.0
        for user in train:  # test / train
            print(user)
            if user not in test:
                continue
            all_items = all_items | set(train[user])
            tu = test[user]
            rank = self.Recommend(user, N)
            for item, _ in rank:
                if item in tu:
                    hit += 1
                popularity += math.log(1 + self.item_pop[item])
                recommend_items.add(item)
            num_rank += len(rank)
            num_tu += len(tu)
        print(hit / (num_rank * 1.0), hit / (num_tu * 1.0), len(recommend_items) / (len(all_items) * 1.0), popularity / (num_rank * 1.0))

        return hit / (num_rank * 1.0), hit / (
            num_tu * 1.0), len(recommend_items) / (len(all_items) * 1.0), popularity / (num_rank * 1.0)

    def Recommend(self, user, N):
        start = time.time()
        rank = dict()
        tagged_items = self.user_items[user]
        ut = self.user_tags[user].keys()  # user 打过的标签
        #  utn = self.user_tags[user].values()  # user 打标签的次数
        for i, tn in self.item_tags.items():  # i: 物品, tn: 标签， 次数
            if i in tagged_items:
                continue
            rank[i] = 0
            for t, n in tn.items():
                if t in ut:
                    rank[i] += n * self.user_tags[user][t]
        rank = sorted(rank.items(), key=lambda d: d[1], reverse=True)[0:N]  # list[(id, rating),()]
        end = time.time()
        print('Recommend %.2fs' % (end - start))
        return rank
        #  return sorted(rank.items(), key=lambda d: d[1], reverse=True)[0:N]  # list[(id, rating),()]

    def main(self):
        path = '~/file/rs/dataset/delicious/user_taggedbookmarks-timestamps.dat'
        data = self.read_data(path, '\t')
        self.train, self.test, self.user_items, self.user_tags, self.item_tags, self.item_pop = self.split_data(data, self.M)
        self.evaluation(self.train, self.test, 10)


if __name__ == '__main__':
    tag = Tag()
    tag.main()
