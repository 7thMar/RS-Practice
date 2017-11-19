#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
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

a = np.random.choice(np.arange(5), (1, 5), replace=False)
#  print(np.arange(5).repeat(3, axis=1))
d = pd.DataFrame(np.zeros((3, 5), dtype=np.int64), index=[1, 2, 3])
for i in d.index:
    d.loc[i, :] = np.random.choice(np.arange(5), (1, 5), replace=False)
#  new = np.arange(5)
#  np.random.shuffle(new)
#  step = np.append(step, new).reshape(i, m)
d.loc[:, 4] = d.loc[:, 0]
d.at[0, 0] = -1
d.loc[:, 5] = d.loc[:, 0]
b, c = d.iloc[0].copy(), d.iloc[1].copy()
d.iloc[0], d.iloc[1] = c, b
arr = np.arange(0, 10, 2)
print(d)

e = [[1, 2, 3], [1, 2, 3]]
f = pd.DataFrame(e, index=[1, 2])
print(f)
