#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.font_manager as f
cnfont = f.FontProperties(
    fname='/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc')
matplotlib.rcParams['axes.unicode_minus'] = False

# plt.title('中文示例', fontproperties=cnfont)


class DataDistribution(object):
    path1k = '~/file/rs/dataset/lastfm-1k/lastfm-dataset-1K/userid-profile.tsv'
    path360k = '~/file/rs/dataset/lastfm-360k/lastfm-dataset-360K/usersha1-profile.tsv'

    def __init__(self):
        pass

    def ReadAndSplitData(self):
        self.data = pd.read_csv(
            self.path360k, sep='\t', usecols=[0, 1, 2]).rename(
                columns={'#id': 'id'})
        gender_data= self.data[True - self.data['gender'].isnull()]
        age_data =  self.data[True - self.data['age'].isnull()]

    def MostPopular(self):
        pass

    def GenderMostPopular(self):
        pass

    def AgeMostPopular(self):
        pass

    def CountryMostPopular(self):
        pass

    def DemographicMostPopular(self):
        pass

    def AgeDistributed(self, age, ax):
        #  age.plot(ax=ax, kind='bar', xticks=None)
        x = np.arange(0, 100)
        ax.bar(x, age.tolist())
        ax.set_title('用户年龄分布', fontproperties=cnfont)

    def GenderDistributed(self, male, female, ax):
        ax.pie([male, female], explode=[0, 0.05], labels=['male', 'female'], colors=['c', 'coral'], autopct='%.2f%%')
        ax.set_title('用户性别比例', fontproperties=cnfont)

    def Analysis(self):
        self.ReadAndSplitData()
        male, female = self.data['gender'].value_counts()
        age = self.data['age'].value_counts().sort_index().loc[1: 100]
        fig, axes = plt.subplots(2, 1)
        self.GenderDistributed(male, female, axes[0])
        self.AgeDistributed(age, axes[1])
        axes[0].legend()
        plt.show()


def main():
    dd = DataDistribution()
    dd.Analysis()


if __name__ == '__main__':
    main()
