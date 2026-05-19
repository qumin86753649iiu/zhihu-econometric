---
title: "用过去预测未来——ARIMA"
series: "像普林斯顿微积分一样学计量经济学"
no: 17
phase: "时间序列"
tags: [时间序列, ARIMA, 预测, ACF, PACF]
---

# 用过去预测未来——ARIMA

> 如果我说："明天股票大概率会涨，因为它在涨。"——这是 AR。
> 如果我说："明天股票大概率会涨，因为昨天出了一个大消息还没消停。"——这是 MA。
> 合在一起，加上差分，就是 ARIMA。

---

## 📈 预测，是一个大胆的假设

在第 16 篇我们学了平稳性。一个平稳的时间序列，它的统计性质（均值、方差）不随时间变化。这就给了我们一个 **大胆但合理的假设**：

> 过去的数据中蕴含的模式，可以推广到未来。

也就是说，我们可以用历史数据"教"模型认识某种模式，然后用这个模式去预测未来。这就是时间序列预测的核心思想。

那怎么描述"过去"到"未来"的关系呢？

答案藏在两个字里：**自相关**（Autocorrelation）。

---

## 🧩 从两个直觉出发：AR 和 MA

### 直觉 1：AR——"昨天的我影响今天的我"

你观察股市波动率的时候，可能会发现一个现象：

> 今天波动率高 → 明天大概率也高
> 今天波动率低 → 明天大概率也低

这就是**持续性**。用数学表达就是：

```
今天的值 = 常数 + 0.7 × 昨天的值 + 随机扰动
```

这就是**一阶自回归模型 AR(1)**：

```
y_t = c + φ · y_{t-1} + ε_t
```

这里的 φ 就是"昨天的值对今天的影响有多大"。如果 φ=0.7，意思是昨天 70% 的"状态"会延续到今天。

### 直觉 2：MA——"昨天的冲击今天还在消化"

另一种情况：某天出了一个重大新闻（比如央行的降准公告），市场当天剧烈波动。但消息的冲击效应不会一天就消失。

```
今天的值 = 常数 + 今天的冲击 + 0.3 × 昨天的冲击
```

这就是**一阶移动平均模型 MA(1)**：

```
y_t = c + ε_t + θ · ε_{t-1}
```

注意这个"移动平均"不是滑动平均。MA 说的是**冲击（shock）**的衰减过程——昨天的消息还在影响今天，但影响力度弱了。

### 核心区别一张表

| 特征 | AR (自回归) | MA (移动平均) |
|------|-------------|---------------|
| "记忆"的内容 | 过去的**值** | 过去的**冲击** |
| PACF 表现 | 在 p 阶后**截尾** | 拖尾（缓慢衰减） |
| ACF 表现 | 拖尾（缓慢衰减） | 在 q 阶后**截尾** |
| 直观理解 | 惯性——今天像昨天 | 冲击余波——昨天的消息还在发酵 |

ACF 和 PACF 是后面用来确定 p 和 q 的关键工具，先留个印象就好。

---

## 🔄 差分阶数 d——什么时候该"做差"？

还记得第 16 篇讲过的吗？非平稳序列会让回归结果全是假的。

ARIMA 里的 **I** 代表 **Integrated**（差分整合）。意思是：如果序列不平稳，就做差分，直到它平稳为止。

```
一阶差分：y_t' = y_t - y_{t-1}
二阶差分：y_t'' = y_t' - y_{t-1}' = y_t - 2y_{t-1} + y_{t-2}
```

**d 就是做差分的次数。**

大多数经济和金融序列，d=1 就够了。只有极少数"超级不平稳"的序列才需要 d=2。

怎么判断差分够了？用 ADF 检验（第 16 篇学过的）。

---

## 🎯 ARIMA(p,d,q)——三部曲

把 AR、差分、MA 三个组件拼在一起：

```
ARIMA(p,d,q) = AR(p)  +  I(d)  +  MA(q)
```

拆开来看：
- **p**：用几个过去的值来预测（AR 的阶数）
- **d**：差分的次数
- **q**：用几个过去的冲击来预测（MA 的阶数）

拟合流程就三步：
1. **确定 d**：ADF 检验 → 是否需要差分 → 差分后再次检验直到平稳
2. **确定 p 和 q**：看差分后序列的 ACF 和 PACF 图
3. **拟合 + 预测**：用选好的 (p,d,q) 跑模型

---

## 🐍 Python 实战——模拟股票波动率预测

### 第一步：模拟一个 ARIMA(1,1,1) 过程

我们假装自己是上帝，先造一个"真实的"数据生成过程：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

np.random.seed(42)

# 真实验参数（上帝视角）
phi = 0.7    # AR(1) 系数——过去值的影响
theta = 0.3  # MA(1) 系数——过去冲击的影响

# 模拟 ARIMA(1,1,1)
n = 300
# 先模拟差分序列（ARMA）
eps = np.random.normal(0, 1, n + 1000)
diff_series = np.zeros(n + 1000)
for t in range(1, len(diff_series)):
    diff_series[t] = phi * diff_series[t-1] + eps[t] + theta * eps[t-1]

diff_series = diff_series[-n:]  # 丢弃 burn-in
y = np.cumsum(diff_series)      # 累积求和→原始序列

# 分割训练/测试
n_train = 240
train = y[:n_train]
test = y[n_train:]
```

这样我们就有了一个"真实但模拟"的波动率指数。

![模拟 ARIMA(1,1,1) 时间序列](images/017-arima-series.png)
*图 1：模拟的 ARIMA(1,1,1) 序列。前 240 个点为训练集（蓝色），后 60 个点为测试集（红色），虚线为分割线。注意序列有明显的趋势性和随机性——这正是金融数据的特征。*

### 第二步：确定 d——ADF 检验

```python
# 原始序列
adf_raw = adfuller(train, autolag='AIC')
print(f"原始序列 ADF p值: {adf_raw[1]:.6f}")

# 差分序列
adf_diff = adfuller(diff_series[:n_train], autolag='AIC')
print(f"一阶差分后 ADF p值: {adf_diff[1]:.6f}")
```

输出大概长这样：

```
原始序列 ADF p值:   0.5832  →  ❌ 非平稳，需要差分
一阶差分后 ADF p值: 0.0000  →  ✅ 平稳
```

结论：**d = 1**。

### 第三步：确定 p 和 q——ACF 和 PACF 是侦探工具

拟合之前，先画两幅图：**ACF（自相关函数）** 和 **PACF（偏自相关函数）**。

- **ACF**：衡量 y_t 和 y_{t-k} 之间的总体相关性（包含了中间路径的影响）
- **PACF**：衡量 y_t 和 y_{t-k} 之间的**净**相关性（剔除了中间路径的影响）

怎么读？

两条黄金规则：
- **PACF 在 p 阶后突然截尾（掉到蓝色区间内）** → AR 的阶数 p 就在那
- **ACF 在 q 阶后突然截尾** → MA 的阶数 q 就在那

如果 ACF 和 PACF 都是拖尾（缓慢衰减），那可能是 ARMA 模型，需要更高阶。

![ACF 和 PACF 对比](images/017-acf-pacf.png)
*图 2：上一排是原始序列（非平稳）的 ACF 和 PACF——ACF 缓慢衰减是典型非平稳信号。下一排是一阶差分后的 ACF 和 PACF——这才是我们用来决定 p 和 q 的图。注意差分后的 PACF 在 lag=1 处显著（超出蓝色区间），之后截尾→p=1；ACF 也在 lag=1 之后截尾→q=1。*

看图说话：

PACF 在 lag=1 处有一个显著的尖峰，之后都落在蓝色置信区间内 → 截尾 → **p=1**

ACF 也在 lag=1 处有一个显著的尖峰，之后截尾 → **q=1**

所以初步确定模型为 **ARIMA(1,1,1)**。

### 第四步：拟合模型

```python
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())
```

输出会显示拟合的系数——它们应该接近我们的"上帝参数" φ=0.7 和 θ=0.3。

### 第五步：预测——验货的时候到了

```python
# 预测测试集的 60 天
forecast_result = model_fit.get_forecast(steps=60)
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int(alpha=0.05)
```

![预测 vs 真实值](images/017-forecast.png)
*图 3：蓝色是训练集历史，红色是测试集真实值，绿色虚线是 ARIMA(1,1,1) 的预测值，绿色阴影是 95% 置信区间。预测的整体趋势跟得上真实值，但置信区间随预测步数增加而扩大——预测越远越不确定，这完全合理。*

### 第六步：评估预测效果

```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(test, forecast_mean)
rmse = np.sqrt(np.mean((test - forecast_mean)**2))
print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}")
```

预测不是魔法。时间越长，误差越大——置信区间的"喇叭形"已经告诉我们了。

---

## 💡 模型诊断——你的模型真的好吗？

拟合完模型，别忘了检查残差。

**残差（residuals） = 实际值 - 拟合值**

好模型的残差应该是**白噪声**——纯随机，没有模式。

怎么检查？**Ljung-Box 检验**：

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

lb = acorr_ljungbox(model_fit.resid, lags=[10, 20, 30], return_df=True)
print(lb)
```

原假设：残差是白噪声。如果 p 值 > 0.05，恭喜，你的模型已经把数据中的"信号"榨干了。

---

## 🔗 延伸：选 p 和 q 不光靠眼睛

ACF/PACF 看图定阶，需要一点经验。更自动化的方法是**遍历多组 (p,q)**，选 AIC 或 BIC 最小的。

```python
results = []
for p in range(0, 4):
    for q in range(0, 4):
        try:
            m = ARIMA(train, order=(p, 1, q)).fit()
            results.append({'p': p, 'q': q, 'aic': m.aic})
        except:
            pass
```

这样你就有了一个"AIC 排行榜"——**哪个组合的 AIC 最小，哪个就是好模型**。

不过注意：数据量特别大的时候，AIC 容易选出过复杂的模型。**BIC 惩罚更狠，对小样本更友好。**

---

## ✅ 小结

这一篇我们学到：

- **AR**（自回归）：用过去的值预测今天——明天像今天
- **MA**（移动平均）：用过去的冲击预测今天——昨天的消息还在发酵
- **差分**：处理非平稳序列的通用武器
- **ARIMA(p,d,q)**：AR + 差分 + MA 的三合一模型
- **ACF 和 PACF**：看图定阶的侦探工具
- **模型诊断**：Ljung-Box 检验残差是否为白噪声
- **预测**：总是带置信区间的——预测越远越不确定

**下一篇预告**：波动率的脾气——GARCH。ARIMA 只关心均值，GARCH 关心方差。金融世界的"波动聚集"背后，就是 GARCH 在管的逻辑。

---

### 🤔 思考题（欢迎评论区讨论）

1. 图 2 中，原始序列的 ACF 呈现缓慢衰减，而差分后的 ACF 快速截尾——你能用自己的话解释为什么吗？
2. 假如你预测一个序列，发现残差的 Ljung-Box 检验 p=0.02（<0.05），这意味着什么？你下一步会怎么做？

---
*下一篇见。均值预测完了，波动率还在闹——GARCH 来了。*
