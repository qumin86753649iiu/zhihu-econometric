"""
知乎计量经济学系列 — 第 15 篇配套代码
主题：趋势与季节——拆解一个时序
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

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
np.random.seed(42)

# ========== 1. 模拟电商月销售额数据 ==========
# 时间轴：2019年1月 ~ 2022年12月 (48个月)
months = pd.date_range(start='2019-01-01', periods=48, freq='ME')
t = np.arange(48)

# --- 趋势分量：线性增长 + 加速 (反映店铺扩张) ---
trend = 100 + 3.5 * t + 0.04 * t**2

# --- 季节分量：12个月周期 ---
# 典型的电商季节模式：2月低（春节）、6月促销、11-12月双11/双12爆发
seasonal_pattern = 20 * np.sin(2 * np.pi * t / 12 + 1.2) + 10 * np.sin(2 * np.pi * t / 6 + 0.5)
# 进一步增强年末促销效应
for i in range(48):
    month = i % 12
    if month == 10:  # 11月 (0-indexed, 10=Nov)
        seasonal_pattern[i] += 25
    elif month == 11:  # 12月
        seasonal_pattern[i] += 20
    elif month == 5:   # 6月促销
        seasonal_pattern[i] += 15
    elif month == 1:   # 2月春节低点
        seasonal_pattern[i] -= 15

# --- 噪声分量 ---
noise = np.random.normal(0, 8, 48)

# --- 合成：加法模型 ---
sales_add = trend + seasonal_pattern + noise

# --- 合成：乘法模型 ---
# 对乘法模型，趋势是基线，季节和噪声是比例因子
sales_mul = trend * (1 + seasonal_pattern / np.mean(trend) * 0.6 + noise / np.mean(trend) * 0.15)

df = pd.DataFrame({
    'date': months,
    'sales_add': sales_add,
    'sales_mul': sales_mul,
    'trend': trend,
    'seasonal': seasonal_pattern,
    'noise': noise
})
df = df.set_index('date')

print("=" * 60)
print("模拟电商月销售额数据（2019-2022）")
print(f"  数据范围: {df.index[0].strftime('%Y-%m')} ~ {df.index[-1].strftime('%Y-%m')}")
print(f"  样本量: {len(df)} 个月")
print(f"  月均销售额 (加法): {df['sales_add'].mean():.1f}")
print(f"  月均销售额 (乘法): {df['sales_mul'].mean():.1f}")
print("=" * 60)

# ===== 图 1：原始时间序列 =====
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(df.index, df['sales_add'], color=C_MAIN, linewidth=2,
        marker='o', markersize=4, markerfacecolor='white',
        markeredgecolor=C_MAIN, markeredgewidth=0.8, label='月销售额')

# 标注季节特征
ax.annotate('春节低点', xy=(pd.Timestamp('2019-02-28'), df.loc['2019-02-28', 'sales_add']),
            xytext=(pd.Timestamp('2019-02-28'), df.loc['2019-02-28', 'sales_add'] - 45),
            fontsize=9, color=C_PURPLE, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_PURPLE, lw=1.2))

ax.annotate('6月促销', xy=(pd.Timestamp('2019-06-30'), df.loc['2019-06-30', 'sales_add']),
            xytext=(pd.Timestamp('2019-06-30'), df.loc['2019-06-30', 'sales_add'] + 30),
            fontsize=9, color=C_GREEN, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=1.2))

ax.annotate('双11/双12\n年末高峰', xy=(pd.Timestamp('2021-11-30'), df.loc['2021-11-30', 'sales_add']),
            xytext=(pd.Timestamp('2021-07-31'), df.loc['2021-07-31', 'sales_add'] + 55),
            fontsize=9, color=C_RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2))

ax.set_xlabel('时间', fontsize=12)
ax.set_ylabel('月销售额（万元）', fontsize=12)
ax.set_title('图 1：某电商平台月销售额（2019-2022）', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/015-raw-series.png', dpi=200, bbox_inches='tight')
print("✅ 图 1 已保存：原始时间序列")

# ========== 2. Seasonal Decompose (加法模型) ==========
decomp_add = seasonal_decompose(df['sales_add'], model='additive', period=12)

# ===== 图 2：分解图 =====
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# (a) 原始序列
axes[0].plot(df.index, decomp_add.observed, color=C_MAIN, linewidth=2)
axes[0].set_ylabel('观测值', fontsize=11)
axes[0].set_title('（a）原始序列 Observed', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.2)

# (b) 趋势
axes[1].plot(df.index, decomp_add.trend, color=C_RED, linewidth=2.5, label='趋势')
axes[1].plot(df.index, trend, color=C_GRAY, linewidth=1.5, linestyle='--',
             alpha=0.5, label='真实趋势（已知）')
axes[1].set_ylabel('趋势', fontsize=11)
axes[1].set_title('（b）趋势分量 Trend', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.2)

# (c) 季节
axes[2].plot(df.index, decomp_add.seasonal, color=C_GREEN, linewidth=2)
axes[2].set_ylabel('季节', fontsize=11)
axes[2].set_title('（c）季节分量 Seasonal', fontsize=12, fontweight='bold')
axes[2].grid(alpha=0.2)

# (d) 残差
axes[3].plot(df.index, decomp_add.resid, color=C_PURPLE, linewidth=1.5,
             marker='o', markersize=2.5, alpha=0.7)
axes[3].axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
axes[3].set_ylabel('残差', fontsize=11)
axes[3].set_xlabel('时间', fontsize=12)
axes[3].set_title('（d）残差分量 Residual', fontsize=12, fontweight='bold')
axes[3].grid(alpha=0.2)

plt.suptitle('图 2：时序加法分解——Observed = Trend + Seasonal + Residual', 
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/015-decomposition.png', dpi=200, bbox_inches='tight')
print("✅ 图 2 已保存：加法分解图")

# ========== 3. 加法 vs 乘法对比 ==========
decomp_add_check = seasonal_decompose(df['sales_add'], model='additive', period=12)
decomp_mul_check = seasonal_decompose(df['sales_mul'], model='multiplicative', period=12)

# ===== 图 3：对比图 =====
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# 第一行：加法分解
# Observed
axes[0, 0].plot(df.index, decomp_add_check.observed, color=C_MAIN, linewidth=1.8)
axes[0, 0].set_title('加法：原始序列', fontsize=12, fontweight='bold')
axes[0, 0].grid(alpha=0.2)

# Trend
axes[0, 1].plot(df.index, decomp_add_check.trend, color=C_RED, linewidth=2.5, label='加法趋势')
axes[0, 1].plot(df.index, decomp_mul_check.trend, color=C_PURPLE, linewidth=1.5, 
                linestyle='--', alpha=0.6, label='乘法趋势（对照）')
axes[0, 1].set_title('加法：趋势分量', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.2)

# Seasonal
axes[0, 2].plot(df.index, decomp_add_check.seasonal, color=C_GREEN, linewidth=2)
axes[0, 2].axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
axes[0, 2].set_title('加法：季节分量', fontsize=12, fontweight='bold')
axes[0, 2].grid(alpha=0.2)

# 第二行：乘法分解
axes[1, 0].plot(df.index, decomp_mul_check.observed, color=C_MAIN, linewidth=1.8)
axes[1, 0].set_title('乘法：原始序列', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.2)

axes[1, 1].plot(df.index, decomp_mul_check.trend, color=C_PURPLE, linewidth=2.5)
axes[1, 1].set_title('乘法：趋势分量', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.2)

axes[1, 2].plot(df.index, decomp_mul_check.seasonal, color=C_ACCENT, linewidth=2)
axes[1, 2].axhline(y=1, color='gray', linewidth=0.5, linestyle='--')
axes[1, 2].set_title('乘法：季节分量（围绕1波动）', fontsize=12, fontweight='bold')
axes[1, 2].grid(alpha=0.2)

# 分析摘要文本
summary_text = (
    "加法模型：Observed = Trend + Seasonal + Residual\n"
    "季节分量以固定振幅（±常数）波动\n"
    "适合季节幅度不随趋势变化的序列\n\n"
    "乘法模型：Observed = Trend × Seasonal × Residual\n"
    "季节分量以比例因子（围绕1）波动\n"
    "适合季节幅度随趋势增长的序列"
)
fig.text(0.5, 0.01, summary_text, ha='center', fontsize=11,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.15))

plt.suptitle('图 3：加法模型 vs 乘法模型分解对比', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(f'{IMAGE_DIR}/015-additive-vs-multiplicative.png', dpi=200, bbox_inches='tight')
print("✅ 图 3 已保存：加法 vs 乘法对比图")

# ========== 4. 分解诊断：残差统计 ==========
resid_add = decomp_add_check.resid.dropna()
resid_mul = decomp_mul_check.resid.dropna()

print("\n" + "=" * 60)
print("分解残差诊断：")
print(f"  加法模型 - 残差标准差: {resid_add.std():.2f}")
print(f"  加法模型 - 残差均值:   {resid_add.mean():.2f}")
print(f"  乘法模型 - 残差标准差: {resid_mul.std():.4f}")
print(f"  乘法模型 - 残差均值:   {resid_mul.mean():.4f}")
print("=" * 60)

# ========== 5. 延伸：简单预测示例 ==========
# 用最后12个月的季节分量外推
last_seasonal = decomp_add_check.seasonal[-12:].values
trend_last = decomp_add_check.trend.dropna().iloc[-1]
trend_slope = np.mean(np.diff(decomp_add_check.trend.dropna().values[-6:]))

print("\n" + "=" * 60)
print("延伸：基于分解的简单预测")
print(f"  最近趋势值: {trend_last:.1f}")
print(f"  趋势斜率: {trend_slope:.2f} /月")
print("  未来12个月预测（线性趋势延伸 + 季节重复）:")
for i in range(12):
    t_next = trend_last + trend_slope * (i + 1)
    s_next = last_seasonal[i]
    pred = t_next + s_next
    print(f"    第 {i+1:2d} 个月: {pred:.1f} = 趋势({t_next:.1f}) + 季节({s_next:.1f})")
print("=" * 60)

print("\n🎉 第 15 篇所有图表已生成完毕！")
