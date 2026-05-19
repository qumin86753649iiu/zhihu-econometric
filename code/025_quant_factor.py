"""
知乎计量经济学系列 — 第 25 篇配套代码
主题：计量+量化选股——用回归造一个选股因子
方法：Fama-MacBeth 两步法, 分层回测, IC 分析
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import spearmanr, t as stats_t
import warnings
warnings.filterwarnings('ignore')

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

print("=" * 60)
print("第 25 篇：计量+量化——用回归造一个选股因子")
print("=" * 60)

# ========== 1. 模拟面板数据 ==========
print("\n>>> 正在模拟数据...")
N, T = 500, 24  # 500 只股票，24 个月

# 真实的因子暴露（每个股票一个，不随时间变化）
beta_true = np.random.normal(0, 1, N)

# 因子收益率（时间序列，每月一个值）
f_t = np.random.normal(0.8, 0.5, T)

# 构建可观测的特征（含噪声）
returns = np.zeros((N, T))
X_char = np.zeros((N, T))
for i in range(N):
    for t in range(T):
        returns[i, t] = 0.5 + beta_true[i] * f_t[t] + np.random.normal(0, 2.0)
        X_char[i, t] = beta_true[i] + np.random.normal(0, 0.5)

print(f"   股票数 N = {N}, 月份 T = {T}")
print(f"   因子月均真实收益: {f_t.mean():.4f}")
print(f"   收益矩阵形状: {returns.shape}")

# ========== 2. Fama-MacBeth 两步法 ==========
print("\n>>> 运行 Fama-MacBeth 两步法...")

# 第一步：每月跑横截面回归
betas = []
for t in range(T):
    X_t = sm.add_constant(X_char[:, t])
    m = sm.OLS(returns[:, t], X_t).fit()
    betas.append(m.params[1])
betas = np.array(betas)

# 第二步：时间序列平均
beta_mean = betas.mean()
beta_se = betas.std(ddof=1) / np.sqrt(T)
t_stat = beta_mean / beta_se
p_val = 2 * (1 - stats_t.cdf(abs(t_stat), df=T - 1))

print(f"   Fama-MacBeth 结果:")
print(f"     因子月均收益 (β) = {beta_mean:.4f}")
print(f"     标准误 (SE)      = {beta_se:.4f}")
print(f"     t 统计量         = {t_stat:.2f}")
print(f"     p 值             = {p_val:.4f}")
print(f"     结论: {'✅ 因子显著' if p_val < 0.05 else '❌ 因子不显著'}")

# ========== 3. 分层回测 ==========
print("\n>>> 运行分层回测...")

def quintile_portfolios(factor, ret):
    """按因子值分 5 组，计算每组每月等权收益"""
    N, T = ret.shape
    port_ret = np.zeros((5, T))
    for t in range(T):
        f_t = factor[:, t]
        r_t = ret[:, t]
        q = pd.qcut(f_t, 5, labels=False)
        for qi in range(5):
            mask = (q == qi)
            port_ret[qi, t] = r_t[mask].mean()
    return port_ret

port_ret = quintile_portfolios(X_char, returns)

# 多空组合 (Q5 - Q1)
ls_ret = port_ret[4] - port_ret[0]
cumulative = (1 + port_ret.T).cumprod(axis=0)
cumulative_ls = (1 + ls_ret).cumprod()

ls_mean = ls_ret.mean()
ls_std = ls_ret.std()
ls_sharpe = ls_mean / ls_std * np.sqrt(12)

print(f"   多空组合月均收益: {ls_mean:.4f}")
print(f"   多空组合月波动率: {ls_std:.4f}")
print(f"   多空组合年化夏普: {ls_sharpe:.2f}")

# ========== 图 1：分层组合累计收益 ==========
print("\n>>> 绘制分层组合累计收益图...")
fig, ax1 = plt.subplots(figsize=(12, 7))

months = np.arange(T)
labels = ['Q1 (最低)', 'Q2', 'Q3', 'Q4', 'Q5 (最高)']
colors = [C_GRAY, C_MAIN, C_GREEN, C_PURPLE, C_ACCENT]

for qi in range(5):
    ax1.plot(months, cumulative[:, qi],
             color=colors[qi], linewidth=2.0,
             label=f'{labels[qi]}')

ax1.fill_between(months, 1, cumulative_ls,
                 color=C_RED, alpha=0.15, label='__nolabel__')
ax1.plot(months, cumulative_ls, color=C_RED,
         linewidth=2.5, linestyle='--', alpha=0.9,
         label=f'Q5-Q1 多空 (累计×{cumulative_ls[-1]:.2f})')

# 标注最后的累计收益
for qi in range(5):
    ax1.annotate(f'{cumulative[-1, qi]:.2f}',
                 (months[-1], cumulative[-1, qi]),
                 textcoords='offset points', xytext=(8, 0),
                 fontsize=9, fontweight='bold', color=colors[qi])

ax1.axhline(y=1.0, color='gray', linewidth=0.8, linestyle=':')
ax1.set_xlabel('月份', fontsize=12)
ax1.set_ylabel('累计收益 (1 = 初始)', fontsize=12)
ax1.set_title('图 1：分层组合累计收益——Q5 和 Q1 的差距就是因子的价值', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left', framealpha=0.9)
ax1.set_xticks(range(0, T, 3))
ax1.set_xticklabels([f'M{t+1}' for t in range(0, T, 3)], fontsize=10)
ax1.grid(alpha=0.2)

# 添加夏普比标注
ax1.text(0.98, 0.05,
         f'多空年化夏普: {ls_sharpe:.2f}',
         transform=ax1.transAxes, fontsize=11,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9),
         ha='right', va='bottom')

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/025-factor-portfolio.png', dpi=200, bbox_inches='tight')
print("✅ 图 1 已保存：025-factor-portfolio.png")

# ========== 图 2：Fama-MacBeth 可视化 ==========
print("\n>>> 绘制 Fama-MacBeth 可视化图...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 左子图：某个月的横截面回归示意
t_example = 5
X_t = X_char[:, t_example]
y_t = returns[:, t_example]
X_sm = sm.add_constant(X_t)
m_example = sm.OLS(y_t, X_sm).fit()

ax1.scatter(X_t, y_t, alpha=0.4, color=C_MAIN, s=15)
x_line = np.linspace(X_t.min(), X_t.max(), 100)
y_line = m_example.params[0] + m_example.params[1] * x_line
ax1.plot(x_line, y_line, color=C_RED, linewidth=2.5,
         label=f'第 {t_example+1} 个月\nβ = {m_example.params[1]:.3f}')
ax1.set_xlabel('因子暴露 (特征值)', fontsize=12)
ax1.set_ylabel('股票收益 (%)', fontsize=12)
ax1.set_title(f'(a) 横截面回归——第 {t_example+1} 个月', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.2)

# 右子图：T 个月的因子系数序列
months_range = np.arange(1, T + 1)
ax2.axhline(y=0, color='gray', linewidth=0.8, linestyle=':')
ax2.bar(months_range, betas, color=[C_GREEN if b > 0 else C_RED for b in betas],
        alpha=0.7, edgecolor='white', linewidth=0.5)
ax2.axhline(y=beta_mean, color=C_MAIN, linewidth=2.5, linestyle='--',
            label=f'均值 β = {beta_mean:.3f}')
# 标注显著区间
ci = beta_se * stats_t.ppf(0.975, T - 1)
ax2.fill_between(months_range,
                 beta_mean - ci,
                 beta_mean + ci,
                 color=C_MAIN, alpha=0.12,
                 label=f'95% CI [{beta_mean-ci:.3f}, {beta_mean+ci:.3f}]')
ax2.set_xlabel('月份', fontsize=12)
ax2.set_ylabel('因子系数 β̂ₜ', fontsize=12)
ax2.set_title(f'(b) 每月因子系数序列 (t = {t_stat:.2f}, p = {p_val:.4f})',
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.set_xticks(range(0, T, 3))
ax2.set_xticklabels([f'M{t+1}' for t in range(0, T, 3)])
ax2.grid(alpha=0.2)

plt.suptitle('图 2：Fama-MacBeth 两步法示意', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/025-fama-macbeth.png', dpi=200, bbox_inches='tight')
print("✅ 图 2 已保存：025-fama-macbeth.png")

# ========== 图 3：IC 分析 ==========
print("\n>>> 绘制 IC 分析图...")

# 计算每月 IC (Spearman 秩相关)
ics = np.array([spearmanr(X_char[:, t], returns[:, t])[0] for t in range(T)])
mean_ic = ics.mean()
ic_std = ics.std()
ir = mean_ic / ic_std
cumulative_ic = np.cumsum(ics)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                gridspec_kw={'height_ratios': [1.2, 1]})

# 上子图：每月 IC 柱状图
colors_ic = [C_GREEN if ic >= 0 else C_RED for ic in ics]
ax1.bar(months_range, ics, color=colors_ic, alpha=0.75,
        edgecolor='white', linewidth=0.5, width=0.7)
ax1.axhline(y=0, color='gray', linewidth=0.8, linestyle=':')
ax1.axhline(y=mean_ic, color=C_MAIN, linewidth=2, linestyle='--',
            label=f'平均 IC = {mean_ic:.4f}')
# 标注 > 0 的比例
pos_ratio = (ics > 0).mean()
ax1.text(0.02, 0.95, f'IC > 0 比例: {pos_ratio:.0%}',
         transform=ax1.transAxes, fontsize=11, va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax1.set_ylabel('IC (Spearman 秩相关)', fontsize=12)
ax1.set_title('图 3a：每月 IC——因子预测精度的时间序列', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_xticks(range(0, T, 3))
ax1.set_xticklabels(range(0, T, 3))
ax1.grid(alpha=0.2)

# 下子图：累计 IC
ax2.plot(months_range, cumulative_ic, color=C_MAIN, linewidth=2.5,
         marker='o', markersize=4, label='累计 IC')
ax2.fill_between(months_range, 0, cumulative_ic,
                 color=C_MAIN, alpha=0.12)
# 标注终值
ax2.annotate(f'{cumulative_ic[-1]:.2f}',
             (months_range[-1], cumulative_ic[-1]),
             textcoords='offset points', xytext=(8, 5),
             fontsize=11, fontweight='bold', color=C_MAIN)
ax2.axhline(y=0, color='gray', linewidth=0.8, linestyle=':')
ax2.set_xlabel('月份', fontsize=12)
ax2.set_ylabel('累计 IC', fontsize=12)
ax2.set_title(f'图 3b：累计 IC——方向一致性检验 (IR = {ir:.2f})',
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_xticks(range(0, T, 3))
ax2.set_xticklabels(range(0, T, 3))
ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/025-ic-analysis.png', dpi=200, bbox_inches='tight')
print("✅ 图 3 已保存：025-ic-analysis.png")

# ========== 5. 汇总报告 ==========
print("\n" + "=" * 60)
print("因子绩效汇总")
print("=" * 60)
print(f"""
    ┌─────────────────────────────────────┐
    │  Fama-MacBeth                        │
    │    β = {beta_mean:.4f}  (t = {t_stat:.2f})           │
    │    p = {p_val:.4f}  {'✅ 显著' if p_val < 0.05 else '❌ 不显著'}              │
    ├─────────────────────────────────────┤
    │  分层回测                            │
    │    多空月均收益: {ls_mean:.4f}                    │
    │    多空年化夏普: {ls_sharpe:.2f}                     │
    ├─────────────────────────────────────┤
    │  IC 分析                             │
    │    平均 IC: {mean_ic:.4f}                           │
    │    IC 标准差: {ic_std:.4f}                        │
    │    IR (信息比): {ir:.2f}                            │
    │    IC > 0 比例: {pos_ratio:.0%}                        │
    └─────────────────────────────────────┘
""")

# ========== 各分层的描述性统计 ==========
print("\n各分层月均收益:")
print(f"{'分层':<10} {'月均收益':<10} {'月波动率':<10}")
print("-" * 32)
for qi in range(5):
    mean_r = port_ret[qi].mean()
    std_r = port_ret[qi].std()
    print(f"{labels[qi]:<10} {mean_r:<10.4f} {std_r:<10.4f}")
print(f"{'Q5-Q1':<10} {ls_mean:<10.4f} {ls_std:<10.4f}")

print("\n" + "=" * 60)
print("🎉 第 25 篇全部图表已生成完毕！")
print("=" * 60)
