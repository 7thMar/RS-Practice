#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ctypes as c
import math
import numpy as np
from multiprocessing import Process, Queue, Pool
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as f
cnfont = f.FontProperties(
    fname='/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc')
matplotlib.rcParams['axes.unicode_minus'] = False
# plt.title('中文示例', fontproperties=cnfont)

a = [1, 2, 3]
b = [3, 5, 4]

d = pd.DataFrame([a, b])
print(d)

print(d.loc[0].sort_values())
