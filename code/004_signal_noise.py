"""
知乎计量经济学系列 — 第 4 篇配套代码
主题：信号 vs 噪声 — t 值、p 值的直觉
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
C_MAIN = '#2E86AB'
C_ACCENT = '#F18F01'
C_RED = '#C73E1D'
C_GREEN = '#4C9F70'

# ========== 1. 纯噪声的模拟：重复实验看 p 值分布 ==========
np.random.seed(42)
n_experiments = 1000
n_obs = 50

p_values = []
t_values = []

for _ in range(n_experiments):
    # 两个完全无关的随机变量
    x = np.random.normal(0, 1, n_obs)
    y = np.random.normal(0, 1, n_obs)
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    p_values.append(model.pvalues[1])
    t_values.append(model.tvalues[1])

p_values = np.array(p_values)
pct_significant = np.mean(p_values < 0.05) * 100
pct_very_significant = np.mean(p_values < 0.01) * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左：p 值分布
ax1 = axes[0]
ax1.hist(p_values, bins=50, color=C_MAIN, alpha=0.7, edgecolor='white', linewidth=0.5)
ax1.axvline(x=0.05, color=C_RED, linewidth=2, linestyle='--', label='p = 0.05')
ax1.axvline(x=0.01, color=C_ACCENT, linewidth=2, linestyle='--', label='p = 0.01')
ax1.set_xlabel('p 值')
ax1.set_ylabel('实验次数')
ax1.set_title(f'1000 次纯噪声实验的 p 值分布\n{pct_significant:.1f}% 显著 (p<0.05) | {pct_very_significant:.1f}% 高度显著 (p<0.01)',
              fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.2)

# 右：t 值分布 + 标准正态对比
ax2 = axes[1]
ax2.hist(t_values, bins=50, color=C_ACCENT, alpha=0.7, edgecolor='white', linewidth=0.5,
         density=True, label='实验 t 值')
x_range = np.linspace(-4, 4, 200)
ax2.plot(x_range, stats.norm.pdf(x_range), color=C_MAIN, linewidth=2.5,
         label='标准正态分布 (理论)')
ax2.axvline(x=1.96, color=C_RED, linewidth=2, linestyle='--', alpha=0.7, label='t = 1.96 (p=0.05)')
ax2.axvline(x=-1.96, color=C_RED, linewidth=2, linestyle='--', alpha=0.7)
ax2.set_xlabel('t 值')
ax2.set_ylabel('密度')
ax2.set_title('t 值分布 vs 理论正态分布', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.2)

plt.suptitle('图1：如果没有真实效应，纯随机也会跑出"显著"结果', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\004-pvalue-simulation.png',
            dpi=200, bbox_inches='tight')
print(f"✅ p 值模拟图已保存 ({pct_significant:.1f}% significant at 0.05)")

# ========== 2. 信号 vs 噪声的对比：两个场景 ==========
np.random.seed(123)

# 场景 A：信号强+噪声小（高 t 值）
n = 30
x = np.random.uniform(1, 10, n)
y_A = 5 + 3.0 * x + np.random.normal(0, 3, n)  # 噪声小

# 场景 B：信号弱+噪声大（低 t 值）
y_B = 5 + 3.0 * x + np.random.normal(0, 20, n)  # 噪声大

model_A = sm.OLS(y_A, sm.add_constant(x)).fit()
model_B = sm.OLS(y_B, sm.add_constant(x)).fit()

t_A = model_A.tvalues[1]
t_B = model_B.tvalues[1]
p_A = model_A.pvalues[1]
p_B = model_B.pvalues[1]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, y, model, label, t_val, p_val in [
    (axes[0], y_A, model_A, '高信噪比', t_A, p_A),
    (axes[1], y_B, model_B, '低信噪比', t_B, p_B)]:
    
    ax.scatter(x, y, alpha=0.6, color=C_MAIN, edgecolors='white', linewidth=0.5, s=50)
    xp = np.linspace(0, 11, 50)
    yp = model.params[0] + model.params[1] * xp
    ax.plot(xp, yp, color=C_RED, linewidth=2.5)
    
    # 残差竖线
    y_pred = model.params[0] + model.params[1] * x
    for i in range(len(x)):
        ax.plot([x[i], x[i]], [y[i], y_pred[i]], color='gray', linewidth=0.5, alpha=0.3)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{label}\nβ₁ = {model.params[1]:.2f} | t = {t_val:.1f} | p = {p_val:.4f}',
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.2)

plt.suptitle('图2：同一个真实效应 (β₁=3)，噪声大小决定了是否"显著"', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\004-signal-noise-comparison.png',
            dpi=200, bbox_inches='tight')
print(f"✅ 信噪比对比图已保存")
print(f"  高信噪比: t={t_A:.1f}, p={p_A:.4f}")
print(f"  低信噪比: t={t_B:.1f}, p={p_B:.4f}")

# ========== 3. 采样分布可视化 ==========
# 从固定 DGP 重复采样，看 β₁ 的分布
np.random.seed(42)
true_beta1 = 2.0
n_reps = 500
n_each = 30

beta_estimates = []

for _ in range(n_reps):
    x_sim = np.random.uniform(1, 10, n_each)
    y_sim = 3 + true_beta1 * x_sim + np.random.normal(0, 8, n_each)
    m = sm.OLS(y_sim, sm.add_constant(x_sim)).fit()
    beta_estimates.append(m.params[1])

beta_estimates = np.array(beta_estimates)

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(beta_estimates, bins=35, color=C_MAIN, alpha=0.7, edgecolor='white', linewidth=0.5,
        density=True, label=f'估计的 β₁ 分布\n均值 = {beta_estimates.mean():.2f}')
ax.axvline(x=true_beta1, color=C_GREEN, linewidth=2.5, linestyle='--', label=f'真实 β₁ = {true_beta1}')

# 标注 ±2 标准误的范围
se = beta_estimates.std()
ax.axvline(x=true_beta1 - 2*se, color=C_RED, linewidth=1.5, linestyle=':', alpha=0.6)
ax.axvline(x=true_beta1 + 2*se, color=C_RED, linewidth=1.5, linestyle=':', alpha=0.6)
ax.fill_betweenx([0, ax.get_ylim()[1]], true_beta1 - 2*se, true_beta1 + 2*se,
                 color=C_RED, alpha=0.06, label='±2 标准误 (约 95% 区间)')

ax.set_xlabel('估计的 β₁')
ax.set_ylabel('密度')
ax.set_title(f'图3：采样分布 — 重复 500 次实验，同一真实效应的估计值分布', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\004-sampling-distribution.png',
            dpi=200, bbox_inches='tight')
print(f"✅ 采样分布图已保存 (SE={se:.3f})")

# ========== 4. 摘要输出 ==========
print(f"""
╔══════════════════════════════════════════════╗
║  信噪比对比                                   ║
╠══════════════════════════════════════════════╣
║  高信噪比:  β₁={model_A.params[1]:.2f}, t={t_A:.1f}, p={p_A:.6f}  ║
║  低信噪比:  β₁={model_B.params[1]:.2f}, t={t_B:.1f}, p={p_B:.4f}  ║
╠══════════════════════════════════════════════╣
║  纯噪声实验: {pct_significant:.1f}% 的 p<0.05                    ║
╚══════════════════════════════════════════════╝
""")
