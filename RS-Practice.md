<h1 id='top'>推荐系统实践</h1>

:frog:

-----

- [推荐系统实践](#top)
  - [利用用户行为数据](#2)
    - [长尾分布](#2-long-tail)
    - [基于用户的协同过滤算法](#2-usercf)
    - [基于物品的协同过滤算法](#2-itemcf)
    - [UserCF & ItemCF 对比](#2-usercf-itemcf)

----
<h2 id='2'>利用用户行为数据</h2>

<h3 id='2-long-tail'>长尾分布</h3>

>  [LongTail.py](https://github.com/7thMar/RS-Practice/blob/master/2-User-Behavior-Data/LongTail.py)


![LongTail](https://github.com/7thMar/RS-Practice/raw/master/2-User-Behavior-Data/image/LongTail-Line.png)


---

<h3 id='2-usercf'>基于用户的协同过滤算法</h3>

> [main.py](https://github.com/7thMar/RS-Practice/blob/master/2-User-Behavior-Data/main.py)
>
> [UserCF.py](https://github.com/7thMar/RS-Practice/blob/master/2-User-Behavior-Data/UserCF.py)

- 用户相似度

  - 余弦相似度

    $$
    w_{uv} = \frac {\vert N(u) \cap N(v) \vert} {\sqrt {\vert N(u) \cap N(v) \vert}}
    $$

  - 倒排表缩短时间

- 用户 *u* 对物品 *i* 的兴趣

    $$
    p_{ui} = \sum_{v \in N(i) \cap S(u, K)} {w_{uv}r_{vi}}
    $$



  > $w_{uv}$: 用户 u 和 v 的相似度
  > $r_{vi}$: 用户 v 对 物品 i 的兴趣(评分)

- 改进 用户相似度

  > 惩罚 *u* *v* 共同兴趣列表热门商品的影响

  $$
    w_{uv} = \frac {\sum_{i \in N(u) \cap N(v)} \frac 1 {\log (1 + \vert N(v) \vert)}} {\sqrt {\vert N(u) \cap N(v) \vert}}
  $$

- 各项指标和K的关系结果
  - 参数
    ```python
    # dataset: MovieLens 1m ratings.dat
    M = 8  # 分组数
    N = 10  # topN
    K = [5, 10, 20, 40, 80, 120, 160]
    ```
  - **Random-MostPopular**
  
    ![Random-MostPopular](https://github.com/7thMar/RS-Practice/raw/master/2-User-Behavior-Data/image/Random-MostPopular-Table.png)

  - **UserCF**
  
    ![UserCF-Table](https://github.com/7thMar/RS-Practice/raw/master/2-User-Behavior-Data/image/UserCF-Table.png)
    
    ![UserCF-Line](https://github.com/7thMar/RS-Practice/raw/master/2-User-Behavior-Data/image/UserCF-Line.png)

  - **UserCF I and II**
  
    ![UserCF-I-II-Table](https://github.com/7thMar/RS-Practice/raw/master/2-User-Behavior-Data/image/UserCF-I-II-Table.png)
    
    ![UserCF-I-II-Line](https://github.com/7thMar/RS-Practice/raw/master/2-User-Behavior-Data/image/UserCF-I-II-Line.png)
    
---

<h3 id='2-itemcf'>基于物品的协同过滤算法</h3>

> [ItemCF.py](https://github.com/7thMar/RS-Practice/blob/master/2-User-Behavior-Data/ItemCF.py)

- 物品相似度
    $$
    w_{ij} = \frac {\vert N(i) \cap N(j) \vert} {\sqrt {\vert N(i) \cap N(j) \vert}}
    $$

- 用户 *u* 对物品 *j* 的兴趣

    $$
    p_{uj} = \sum_{i \in N(u) \cap S(j, K)} {w_{ji}r_{ui}}
    $$



  > $w_{ji}$: 物品 j 和 i 的相似度
  > $r_{ui}$: 用户对 物品 i 的兴趣(评分)

- 各项指标和K的关系结果
  - 参数
    ```python
    # dataset: MovieLens 1m ratings.dat
    M = 8  # 分组数
    N = 10  # topN
    K = [5, 10, 20, 40, 80, 120, 160]
    ```

  ![ItemCF](https://github.com/7thMar/RS-Practice/raw/master/2-User-Behavior-Data/image/ItemCF.png)

---

<h3 id='2-usercf-itemcf'>ItemCF & UserCF 对比结果</h3>

![UserCF & ItemCF 随K值变化的各项指标对比](https://github.com/7thMar/RS-Practice/raw/master/2-User-Behavior-Data/image/UserCF-ItemCF-Line.png)
