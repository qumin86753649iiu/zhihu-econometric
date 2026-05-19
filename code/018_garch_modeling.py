"""
知乎计量经济学系列 — 第 18 篇配套代码
主题：波动率的脾气——GARCH
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from scipy import stats

# ===== 统一风格 =====
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
C_MAIN = '#2E86AB'
C_ACCENT = '#F18F01'
C_RED = '#C73E1D'
C_GREEN = '#4C9F70'
C_PURPLE = '#A23B72'
C_GRAY = '#A0A0A0'
C_LIGHT_BLUE = '#D4E8F0'
C_LIGHT_RED = '#F5D0C5'
IMAGE_DIR = r'C:\Users\qumin\projects\zhihu-econometrics\images'

np.random.seed(42)

print("=" * 55)
print("第 18 篇：波动率的脾气——GARCH")
print("=" * 55)

# ========== 1. 模拟 GARCH(1,1) 过程 ==========
print("\n>>> 正在模拟 GARCH(1,1) 过程...")

n = 2000  # 样本量（约 8 年日数据）
omega = 0.05
alpha = 0.10  # 对最新冲击的反应
beta = 0.85   # 对过去方差的延续

# 检查平稳条件：alpha + beta < 1
print(f"   参数: ω={omega}, α={alpha}, β={beta}")
print(f"   持续性: α+β = {alpha+beta:.2f} {'< 1 ✅ 平稳' if alpha+beta < 1 else '>= 1 ❌ 不平稳'}")

sigma2 = np.zeros(n)
eps = np.zeros(n)
returns = np.zeros(n)

# 初始方差设为无条件方差
sigma2[0] = omega / (1 - alpha - beta)
eps[0] = np.random.normal(0, np.sqrt(sigma2[0]))
returns[0] = eps[0]

# 递归生成
for t in range(1, n):
    sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
    eps[t] = np.random.normal(0, np.sqrt(sigma2[t]))
    returns[t] = eps[t]

df = pd.DataFrame({
    'returns': returns,
    'volatility': np.sqrt(sigma2)
})
df.index = pd.date_range(start='2017-01-01', periods=n, freq='B')

print(f"   生成了 {n} 个交易日的数据")
print(f"   日收益率标准差: {returns.std():.4f}")
print(f"   年化波动率: {returns.std() * np.sqrt(252):.2%}")

# ===== 图 1：收益率与波动率——看"情绪"的聚集效应 =====
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# 上：收益率序列
ax1 = axes[0]
ax1.plot(df.index, df['returns'], color=C_MAIN, linewidth=0.6, alpha=0.8)
# 用颜色填充突出波动聚集区域
# 找出波动率超过 90% 分位数的时期
high_vol_threshold = np.percentile(df['volatility'], 90)
high_vol_periods = df['volatility'] > high_vol_threshold
ax1.fill_between(df.index, df['returns'].min(), df['returns'].max(),
                 where=high_vol_periods, color=C_RED, alpha=0.08, label='高波动期')
ax1.axhline(y=0, color='gray', linewidth=0.5)
ax1.set_ylabel('日收益率', fontsize=12)
ax1.set_title('图 1a：模拟日收益率序列——"涨跌情绪"的聚集效应', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(alpha=0.15)

# 下：真实波动率（潜在）
ax2 = axes[0].twinx()
ax2.plot(df.index, df['volatility'], color=C_RED, linewidth=0.8, alpha=0.5, linestyle='--')
ax2.set_ylabel('条件波动率（隐变量）', fontsize=12, color=C_RED)

# 下：幅度（绝对值）——用于展示波动聚集
ax_b = axes[1]
ax_b.fill_between(df.index, 0, np.abs(df['returns']),
                  color=C_MAIN, alpha=0.5, label='|收益率|（波动代理）')
ax_b.fill_between(df.index, 0, np.abs(df['returns']),
                  where=high_vol_periods, color=C_RED, alpha=0.25)
ax_b.axhline(y=np.abs(df['returns']).mean(), color=C_GREEN, linewidth=1.5,
             linestyle='--', label=f'平均幅度 = {np.abs(df["returns"]).mean():.3f}')
ax_b.set_ylabel('|收益率|', fontsize=12)
ax_b.set_xlabel('时间', fontsize=12)
ax_b.set_title('图 1b：收益率绝对值——波动聚集更明显', fontsize=13, fontweight='bold')
ax_b.legend(fontsize=9, loc='upper right')
ax_b.grid(alpha=0.15)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/018-returns-volatility.png', dpi=200, bbox_inches='tight')
print("✅ 图 1 已保存：收益率与波动聚集图")

# ========== 2. 拟合 GARCH(1,1) ==========
print("\n>>> 正在拟合 GARCH(1,1) 模型...")

# 用 arch_model 拟合
garch_model = arch_model(
    df['returns'] * 100,  # 转为百分比（收敛更稳定）
    vol='Garch',
    p=1,
    q=1,
    mean='Zero',  # 假设均值为 0（金融日收益率的常见假设）
    dist='normal'
)
garch_result = garch_model.fit(disp='off')

print(garch_result.summary())

# 提取结果
params = garch_result.params
fitted_omega = params.get('omega', 0)
fitted_alpha = params.get('alpha[1]', 0)
fitted_beta = params.get('beta[1]', 0)

print(f"\n   拟合参数：")
print(f"     ω = {fitted_omega:.4f}  (真实: {omega})")
print(f"     α = {fitted_alpha:.4f}  (真实: {alpha})")
print(f"     β = {fitted_beta:.4f}  (真实: {beta})")
print(f"     α+β = {fitted_alpha + fitted_beta:.4f}  (持续性的度量)")

# 提取条件波动率（转回原始尺度）
cond_vol = garch_result.conditional_volatility / 100  # 从百分比转回

# ===== 图 2：拟合的条件波动率 vs 真实波动率 =====
fig, ax = plt.subplots(figsize=(14, 6))

# 真实波动率（模拟时已知）
true_vol = df['volatility']

ax.plot(df.index, true_vol, color=C_GRAY, linewidth=1.0,
        alpha=0.5, label='真实波动率（模拟时的隐变量）')
ax.plot(df.index, cond_vol, color=C_RED, linewidth=1.5,
        label=f'GARCH(1,1) 拟合波动率\n(α={fitted_alpha:.3f}, β={fitted_beta:.3f})')

# 标注一些波动聚集区
# 找出拟合的峰值
peak_idx = cond_vol.nlargest(5).index
for idx in peak_idx:
    ax.axvline(x=idx, color=C_RED, alpha=0.15, linewidth=0.8)

ax.set_xlabel('时间', fontsize=12)
ax.set_ylabel('波动率（标准差）', fontsize=12)
ax.set_title('图 2：GARCH(1,1) 拟合的条件波动率 vs 真实波动率', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.15)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/018-garch-fit.png', dpi=200, bbox_inches='tight')
print("✅ 图 2 已保存：GARCH 拟合图")

# ========== 3. VaR 计算与可视化 ==========
print("\n>>> 正在计算 VaR（Value at Risk）...")

# 用 GARCH 预测的波动率计算动态 VaR
confidence_95 = 1.645  # 95% 单侧正态分位数
confidence_99 = 2.326  # 99% 单侧正态分位数

var_95 = -cond_vol * confidence_95
var_99 = -cond_vol * confidence_99

# 统计穿透率
breaches_95 = df['returns'] < var_95
breaches_99 = df['returns'] < var_99
breach_rate_95 = breaches_95.mean() * 100
breach_rate_99 = breaches_99.mean() * 100

print(f"\n   VaR 穿透率检验（期望 = 5% 和 1%）：")
print(f"     95% VaR 穿透率: {breach_rate_95:.2f}%")
print(f"     99% VaR 穿透率: {breach_rate_99:.2f}%")

# ===== 图 3：VaR 可视化 =====
fig, ax = plt.subplots(figsize=(14, 7))

# 收益率主线
ax.plot(df.index, df['returns'], color=C_MAIN, linewidth=0.6,
        alpha=0.7, label='日收益率')

# VaR 带
ax.plot(df.index, var_95, color=C_ACCENT, linewidth=1.5,
        linestyle='--', label=f'95% VaR (z={confidence_95})')
ax.plot(df.index, var_99, color=C_RED, linewidth=1.5,
        linestyle='--', label=f'99% VaR (z={confidence_99})')

# 标注穿透点
# 只显示近 500 个点以便看清楚
plot_start = max(0, n - 800)
for i in range(plot_start, n):
    if breaches_99.iloc[i]:
        ax.scatter(df.index[i], df['returns'].iloc[i],
                   color=C_RED, s=50, zorder=5, marker='o',
                   edgecolors='white', linewidth=0.6)
    elif breaches_95.iloc[i]:
        ax.scatter(df.index[i], df['returns'].iloc[i],
                   color=C_ACCENT, s=30, zorder=5, marker='o',
                   edgecolors='white', linewidth=0.5)

# 填充 95%-99% 区间
ax.fill_between(df.index[plot_start:], var_95[plot_start:], var_99[plot_start:],
                color=C_ACCENT, alpha=0.08, label='95%-99% 尾部区间')

ax.set_xlabel('时间', fontsize=12)
ax.set_ylabel('收益率', fontsize=12)
ax.set_title(f'图 3：基于 GARCH 波动率的动态 VaR（95% 穿透率 {breach_rate_95:.1f}%，99% 穿透率 {breach_rate_99:.1f}%）',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower left')
ax.grid(alpha=0.15)
ax.set_xlim(df.index[plot_start], df.index[-1])

# 嵌入小图：总穿透率统计
from matplotlib.patches import FancyBboxPatch
stats_text = (
    f"VaR 穿透统计（全样本）\n"
    f"95% VaR: {breach_rate_95:.1f}%\n"
    f"99% VaR: {breach_rate_99:.1f}%"
)
ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/018-var-demo.png', dpi=200, bbox_inches='tight')
print("✅ 图 3 已保存：VaR 可视化图")

# ========== 4. 额外分析：VaR 的"时变性" ==========
print("\n>>> 附加分析：静态 VaR vs 动态 VaR 对比...")

# 静态 VaR（假设正态分布，恒定波动率）
static_sigma = df['returns'].std()
static_var_95 = -static_sigma * confidence_95
static_var_99 = -static_sigma * confidence_99

# 动态 VaR 的极端值范围
print(f"\n   静态 95% VaR: {static_var_95:.4f}")
print(f"   动态 95% VaR 范围: [{var_95.min():.4f}, {var_95.max():.4f}]")
print(f"\n   静态 99% VaR: {static_var_99:.4f}")
print(f"   动态 99% VaR 范围: [{var_99.min():.4f}, {var_99.max():.4f}]")
print(f"\n   → 静态度量低估了风险高峰期，高估了平静期。")

print("\n" + "=" * 55)
print("🎉 第 18 篇全部图表已生成完毕！")
print("=" * 55)
