"""
知乎计量经济学系列 — 第 17 篇配套代码
主题：用过去预测未来——ARIMA
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# ===== 统一风格 =====
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
C_MAIN = '#2E86AB'
C_ACCENT = '#F18F01'
C_RED = '#C73E1D'
C_GREEN = '#4C9F70'
C_PURPLE = '#A23B72'
C_GRAY = '#A0A0A0'
IMAGE_DIR = r'C:\Users\qumin\projects\zhihu-econometrics\images'

# 随机种子
np.random.seed(42)

# ========== 1. 模拟一个 ARIMA(1,1,1) 过程 ==========
print("=" * 60)
print("步骤一：模拟 ARIMA(1,1,1) 过程")
print("=" * 60)

n = 300  # 总样本量
n_train = 240  # 训练集长度
n_test = n - n_train  # 测试集长度

# 模拟 ARIMA(1,1,1):
# 先模拟 ARMA(1,1) for the differenced series
# y_t' = φ * y_{t-1}' + ε_t + θ * ε_{t-1}
# 其中 y_t' = y_t - y_{t-1} (一阶差分)

phi = 0.7    # AR(1) 系数
theta = 0.3  # MA(1) 系数

# 生成白噪声
eps = np.random.normal(0, 1, n + 1000)  # 多生成一些来消除初始条件影响

# 生成差分序列 (ARMA(1,1))
diff_series = np.zeros(n + 1000)
for t in range(1, len(diff_series)):
    diff_series[t] = phi * diff_series[t-1] + eps[t] + theta * eps[t-1]

# 丢弃 burn-in 部分
diff_series = diff_series[-n:]

# 累积求和得到原始序列 (ARIMA(1,1,1))
y = np.cumsum(diff_series)
# 加上一点漂移让序列看起来更真实
y = y + np.linspace(0, 5, n)

# 构造 DataFrame
dates = pd.date_range(start='2022-01-01', periods=n, freq='D')
df = pd.DataFrame({
    'value': y,
    'diff': diff_series
}, index=dates)

train = df.iloc[:n_train]
test = df.iloc[n_train:]

print(f"总样本量: {n}")
print(f"训练集: {n_train} ({(n_train/n)*100:.0f}%)")
print(f"测试集: {n_test} ({(n_test/n)*100:.0f}%)")
print(f"AR(1) 系数 φ = {phi}")
print(f"MA(1) 系数 θ = {theta}")

# ===== 图 1：时间序列 + 训练/测试分割 =====
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(train.index, train['value'], color=C_MAIN, linewidth=1.5,
        label=f'训练集 ({n_train} 天)')
ax.plot(test.index, test['value'], color=C_RED, linewidth=1.5,
        label=f'测试集 ({n_test} 天)', linestyle='--')
ax.axvline(x=train.index[-1], color=C_GRAY, linewidth=1.5,
           linestyle=':', alpha=0.7, label='分割点')

ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('模拟价格 / 波动率指数', fontsize=12)
ax.set_title('图 1：模拟 ARIMA(1,1,1) 时间序列——训练集与测试集', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/017-arima-series.png',
            dpi=200, bbox_inches='tight')
print("\n✅ 图 1 已保存：时间序列 + 训练/测试分割")

# ========== 2. 平稳性检验 (ADF) 和差分 ==========
print("\n" + "=" * 60)
print("步骤二：平稳性检验——是否需要差分？")
print("=" * 60)

# 对原始序列做 ADF 检验
adf_raw = adfuller(train['value'], autolag='AIC')
print(f"\n原始序列 ADF 检验:")
print(f"  ADF 统计量: {adf_raw[0]:.4f}")
print(f"  p 值:       {adf_raw[1]:.6f}")
print(f"  结论: {'✅ 平稳' if adf_raw[1] < 0.05 else '❌ 非平稳，需要差分'}")

# 对差分序列做 ADF 检验
adf_diff = adfuller(train['diff'], autolag='AIC')
print(f"\n一阶差分序列 ADF 检验:")
print(f"  ADF 统计量: {adf_diff[0]:.4f}")
print(f"  p 值:       {adf_diff[1]:.6f}")
print(f"  结论: {'✅ 平稳' if adf_diff[1] < 0.05 else '❌ 非平稳'}")

# 确定 d
d = 0 if adf_raw[1] < 0.05 else 1
print(f"\n→ 确定差分阶数 d = {d}")

# ========== 3. ACF 和 PACF —— 确定 p 和 q ==========
print("\n" + "=" * 60)
print("步骤三：ACF & PACF——确定 p 和 q")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 原始序列 ACF/PACF
plot_acf(train['value'], lags=30, ax=axes[0, 0], alpha=0.05,
         color=C_MAIN, title='（a）原始序列 ACF')
axes[0, 0].grid(alpha=0.2)

plot_pacf(train['value'], lags=30, ax=axes[0, 1], alpha=0.05,
          method='ywm', color=C_ACCENT, title='（b）原始序列 PACF')
axes[0, 1].grid(alpha=0.2)

# 差分序列 ACF/PACF
plot_acf(train['diff'], lags=30, ax=axes[1, 0], alpha=0.05,
         color=C_MAIN, title='（c）一阶差分后 ACF')
axes[1, 0].grid(alpha=0.2)

plot_pacf(train['diff'], lags=30, ax=axes[1, 1], alpha=0.05,
          method='ywm', color=C_ACCENT, title='（d）一阶差分后 PACF')
axes[1, 1].grid(alpha=0.2)

plt.suptitle('图 2：ACF 与 PACF——识别 AR 和 MA 的阶数', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/017-acf-pacf.png',
            dpi=200, bbox_inches='tight')
print("\n✅ 图 2 已保存：ACF & PACF 对比图")

# 通过 ACF/PACF 判断 p 和 q
# 差分序列的 PACF 在 lag=1 处显著截尾 → p=1
# 差分序列的 ACF 在 lag=1 处显著截尾 → q=1
p_guess = 1
q_guess = 1
print(f"\n→ 观察差分序列后，初步确定 p ≈ {p_guess}, q ≈ {q_guess}")
print(f"  结合 d = {d}，模型为 ARIMA({p_guess},{d},{q_guess})")

# ========== 4. 手动拟合 ARIMA ==========
print("\n" + "=" * 60)
print(f"步骤四：拟合 ARIMA({p_guess},{d},{q_guess}) 模型")
print("=" * 60)

model = ARIMA(train['value'], order=(p_guess, d, q_guess))
model_fit = model.fit()

print(model_fit.summary())

# ========== 5. 模型诊断 ==========
print("\n" + "=" * 60)
print("步骤五：模型诊断——残差检验")
print("=" * 60)

# 残差分析
residuals = model_fit.resid

# Ljung-Box 检验（残差是否白噪声）
lb_test = sm.stats.diagnostic.acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
print("\nLjung-Box 检验（原假设：残差为白噪声）：")
for lag in [10, 20, 30]:
    if lag in lb_test.index:
        print(f"  lag={lag}: p值 = {lb_test.loc[lag, 'lb_pvalue']:.4f}",
              f"{'✅ 白噪声' if lb_test.loc[lag, 'lb_pvalue'] > 0.05 else '⚠️ 仍有自相关'}")

print("\n拟合参数与真实参数对比：")
print(f"  {'参数':<20} {'真实值':<10} {'估计值':<10} {'偏差':<10}")
print(f"  {'-'*50}")
print(f"  {'AR(1) φ':<20} {phi:<10.3f} {model_fit.arparams[0] if hasattr(model_fit, 'arparams') and len(model_fit.arparams) > 0 else 'N/A':<10} {'N/A':<10}")
if hasattr(model_fit, 'maparams') and len(model_fit.maparams) > 0:
    print(f"  {'MA(1) θ':<20} {theta:<10.3f} {model_fit.maparams[0]:<10.3f} {model_fit.maparams[0] - theta:<10.3f}")
print()

# ========== 6. 预测：训练集上拟合，测试集上预测 ==========
print("=" * 60)
print("步骤六：预测——用训练集拟合，在测试集上验证")
print("=" * 60)

# 方式一：动态预测（用预测值做下一步的输入）
forecast_result = model_fit.get_forecast(steps=n_test)
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int(alpha=0.05)

# 方式二：样本内拟合——对比
fitted_values = model_fit.fittedvalues

# ===== 图 3：预测 vs 真实值 =====
fig, ax = plt.subplots(figsize=(14, 7))

# 训练集
ax.plot(train.index, train['value'], color=C_MAIN, linewidth=1.2,
        label=f'训练集（历史数据）', alpha=0.7)

# 测试集真实值
ax.plot(test.index, test['value'], color=C_RED, linewidth=1.8,
        label='测试集（真实值）', alpha=0.8)

# 预测值
ax.plot(test.index, forecast_mean, color=C_GREEN, linewidth=2.0,
        linestyle='--', marker='o', markersize=4,
        label='ARIMA 预测值')

# 置信区间
ax.fill_between(test.index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1],
                color=C_GREEN, alpha=0.15,
                label='95% 置信区间')

# 分割线
ax.axvline(x=train.index[-1], color=C_GRAY, linewidth=1.5,
           linestyle=':', alpha=0.5)

ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('模拟价格 / 波动率指数', fontsize=12)
ax.set_title('图 3：ARIMA 预测 vs 真实值的较量', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/017-forecast.png',
            dpi=200, bbox_inches='tight')
print("\n✅ 图 3 已保存：预测 vs 真实值")

# ========== 7. 预测误差评估 ==========
print("\n" + "=" * 60)
print("预测误差评估")
print("=" * 60)

# 对齐时间索引
actual = test['value'].values
predicted = forecast_mean.values

# 评估指标
mse = np.mean((actual - predicted) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(actual - predicted))
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

# 方向准确率（方向对了多少？）
actual_dir = np.sign(np.diff(actual))
pred_dir = np.sign(np.diff(predicted))
# Trim to same length if needed
min_len = min(len(actual_dir), len(pred_dir))
dir_acc = np.mean(actual_dir[:min_len] == pred_dir[:min_len]) * 100

print(f"\n  RMSE (均方根误差): {rmse:.3f}")
print(f"  MAE  (平均绝对误差): {mae:.3f}")
print(f"  MAPE (平均绝对百分比误差): {mape:.2f}%")
print(f"  方向准确率: {dir_acc:.1f}%")

# ========== 8. 额外：不同 p/q 组合对比 ==========
print("\n" + "=" * 60)
print("扩展：不同 p,q 组合的 AIC 比较")
print("=" * 60)

results = []
for p in range(0, 4):
    for q in range(0, 4):
        try:
            m = ARIMA(train['value'], order=(p, d, q)).fit()
            results.append({'p': p, 'q': q, 'aic': m.aic, 'bic': m.bic})
        except Exception:
            pass

results_df = pd.DataFrame(results)
if len(results_df) > 0:
    best = results_df.loc[results_df['aic'].idxmin()]
    print(f"\n最佳模型 (按 AIC)：ARIMA({int(best['p'])},{d},{int(best['q'])})")
    print(f"  AIC = {best['aic']:.2f}")
    results_df = results_df.sort_values('aic')
    print("\n各模型 AIC 排名：")
    print(results_df.to_string(index=False))

print("\n" + "=" * 60)
print("🎉 第 17 篇所有图表和数据已生成完毕！")
print("=" * 60)
