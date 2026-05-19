"""
知乎计量经济学系列 — 第 27 篇配套代码
主题：Logit 回归——是"是"还是"否"
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit as sigmoid

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

# ========== 1. 模拟数据：银行贷款违约预测 ==========
np.random.seed(42)
n = 500

# 核心驱动变量：负债率（Debt-to-Income Ratio）
dti = np.random.uniform(0.1, 0.8, n)

# 真实违约概率：负债率越高，违约概率越大（非线性 Logit 形式）
logit_p = -3 + 5 * dti
prob_true = 1 / (1 + np.exp(-logit_p))

# 生成二值结果（0 = 不违约, 1 = 违约）
default = np.random.binomial(1, prob_true)

df = pd.DataFrame({'dti': dti, 'default': default, 'prob_true': prob_true})
print(f"数据形状: {df.shape}")
print(f"违约率: {df['default'].mean():.2%}")
print(f"负债率范围: [{df['dti'].min():.2f}, {df['dti'].max():.2f}]")

# ========== 2. 图 1：LPM vs Logit 对比 ==========
# --- 2a. 线性概率模型 (OLS) ---
X = sm.add_constant(df['dti'])
lpm = sm.OLS(df['default'], X).fit()
df['lpm_fitted'] = lpm.fittedvalues

print("\n" + "=" * 60)
print("线性概率模型 (LPM) - OLS 结果：")
print("=" * 60)
print(lpm.summary().tables[1])

# 检查 LPM 的越界情况
n_out_of_bounds = ((df['lpm_fitted'] < 0) | (df['lpm_fitted'] > 1)).sum()
print(f"\nLPM 预测值越界样本数: {n_out_of_bounds}/{len(df)} ({n_out_of_bounds/len(df):.1%})")

# --- 2b. Logit 模型 ---
logit_mod = smf.logit('default ~ dti', data=df)
logit_res = logit_mod.fit(disp=False)
df['logit_fitted'] = logit_res.predict()

print("\n" + "=" * 60)
print("Logit 回归结果：")
print("=" * 60)
print(logit_res.summary())

# --- 图 1 绘制：LPM vs Logit ---
x_grid = np.linspace(0, 0.9, 300)
x_grid_with_const = sm.add_constant(x_grid)

lpm_line = lpm.predict(x_grid_with_const)
logit_line = logit_res.predict(sm.add_constant(pd.DataFrame({'dti': x_grid})))

fig, ax = plt.subplots(figsize=(10, 6))

# 实际数据点（抖动处理 jitter 避免重叠）
jitter = 0.03
ax.scatter(df['dti'], df['default'] + np.random.uniform(-jitter, jitter, n),
           alpha=0.3, color=C_GRAY, s=15, label='实际数据 (抖动处理)')

# LPM 拟合线
ax.plot(x_grid, lpm_line, color=C_RED, linewidth=2.5, linestyle='--',
        label=f'LPM (OLS): P = {lpm.params.iloc[0]:.3f} + {lpm.params.iloc[1]:.3f}×DTI')

# Logit 拟合线 (S 曲线)
ax.plot(x_grid, logit_line, color=C_MAIN, linewidth=3,
        label='Logit: S 曲线')

# [0,1] 边界参考线
ax.axhline(y=0, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
ax.axhline(y=1, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)

# 标注 LPM 越界区域
ax.annotate('LPM < 0\n(无意义)', xy=(0.05, lpm_line[15]),
            xytext=(0.05, -0.15), fontsize=9, color=C_RED,
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2))

ax.set_xlabel('负债率 (Debt-to-Income Ratio)', fontsize=12)
ax.set_ylabel('违约概率 P(Y=1)', fontsize=12)
ax.set_title('图 1：线性概率模型 vs Logit——为什么 OLS 不适合分类问题',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.set_xlim(0, 0.9)
ax.set_ylim(-0.3, 1.3)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/027-lpm-vs-logit.png', dpi=200, bbox_inches='tight')
print("✅ 图 1 已保存：027-lpm-vs-logit.png")

# ========== 3. 图 2：Logit S 曲线 + 决策边界 ==========
fig, ax = plt.subplots(figsize=(10, 6))

# 真实概率（不含噪声的 S 曲线）
real_line = sigmoid(-3 + 5 * x_grid)
ax.plot(x_grid, real_line, color=C_GREEN, linewidth=2,
        linestyle=':', alpha=0.7, label='真实概率 (DGP)')

# Logit 拟合的 S 曲线
ax.plot(x_grid, logit_line, color=C_MAIN, linewidth=3,
        label='Logit 拟合 S 曲线')

# 决策边界 P=0.5
ax.axhline(y=0.5, color=C_RED, linewidth=1.5, linestyle='--', alpha=0.7)
decision_x = x_grid[np.argmin(np.abs(logit_line - 0.5))]
ax.axvline(x=decision_x, color=C_RED, linewidth=1.5, linestyle='--', alpha=0.7)

# 标注决策边界
ax.annotate(f'决策边界\nDTI ≈ {decision_x:.2f}',
            xy=(decision_x, 0.5), xytext=(decision_x + 0.12, 0.35),
            fontsize=10, color=C_RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.5))

# 标注区域
ax.fill_between(x_grid, 0, 0.5, alpha=0.05, color=C_GREEN, label='预测不违约')
ax.fill_between(x_grid, 0.5, 1, alpha=0.05, color=C_ACCENT, label='预测违约')

# 显示实际数据点
colors = df['default'].map({0: C_GREEN, 1: C_ACCENT})
ax.scatter(df['dti'], df['default'] + np.random.uniform(-0.02, 0.02, n),
           c=colors, alpha=0.2, s=10, edgecolors='none')

ax.set_xlabel('负债率 (DTI)', fontsize=12)
ax.set_ylabel('违约概率', fontsize=12)
ax.set_title('图 2：Logit S 曲线与决策边界', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.set_xlim(0, 0.9)
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/027-logit-curve.png', dpi=200, bbox_inches='tight')
print("✅ 图 2 已保存：027-logit-curve.png")

# ========== 4. 几率比 (Odds Ratio) ==========
odds_ratio = np.exp(logit_res.params)
print("\n" + "=" * 60)
print("几率比 (Odds Ratio)：")
print("=" * 60)
for name, val in odds_ratio.items():
    print(f"  {name:>12}: {val:>8.3f}")

# 每 0.1 个单位的几率比
or_per_01 = np.exp(logit_res.params['dti'] * 0.1)
print(f"\n  dti (每 +0.1): {or_per_01:.3f}  (违约几率增加 {(or_per_01-1)*100:.1f}%)")

# ========== 5. 边际效应 (Marginal Effects at Mean) ==========
print("\n" + "=" * 60)
print("边际效应——在均值处 (MEM)：")
print("=" * 60)
margeff = logit_res.get_margeff(at='mean')
print(margeff.summary())

# 手动验证 MEM 计算
# 在均值处的边际效应 = β × P×(1-P)
dti_mean = df['dti'].mean()
p_at_mean = logit_res.predict(pd.DataFrame({'dti': [dti_mean]}))
me_manual = logit_res.params['dti'] * p_at_mean.iloc[0] * (1 - p_at_mean.iloc[0])
print(f"\n手动验证 MEM:")
print(f"  dti 均值 = {dti_mean:.3f}")
print(f"  均值处概率 P = {p_at_mean.iloc[0]:.3f}")
print(f"  边际效应 (手动) = β × P × (1-P) = {logit_res.params['dti']:.3f} × {p_at_mean.iloc[0]:.3f} × {1-p_at_mean.iloc[0]:.3f} = {me_manual:.3f}")
print(f"  边际效应 (statsmodels) = {margeff.margeff[0]:.3f}")

# ========== 6. 图 3：边际效应可视化 ==========
# 绘制 S 曲线 + 在均值处的切线（边际效应）
fig, ax = plt.subplots(figsize=(10, 6))

# S 曲线
ax.plot(x_grid, logit_line, color=C_MAIN, linewidth=3, label='Logit 预测概率')

# 均值点
y_at_mean = p_at_mean.iloc[0]
ax.scatter([dti_mean], [y_at_mean], color=C_RED, s=120, zorder=5,
           edgecolors='white', linewidth=2, label=f'均值点 (DTI={dti_mean:.2f})')

# 切线（斜率 = 边际效应）
# 切线: y - y0 = m * (x - x0)
tan_x = np.array([dti_mean - 0.2, dti_mean + 0.2])
tan_y = y_at_mean + me_manual * (tan_x - dti_mean)
ax.plot(tan_x, tan_y, color=C_RED, linewidth=2, linestyle='--', alpha=0.8,
        label=f'切线 (斜率 = {me_manual:.3f})')

# 标注边际效应的含义
ax.annotate(f'边际效应 (MEM) = {me_manual:.3f}\n即 DTI 每+1, P 增加 {me_manual*100:.1f}pp',
            xy=(dti_mean + 0.05, y_at_mean + 0.15),
            fontsize=10, color=C_RED, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# 显示斜率在两端接近于 0
ax.annotate('斜率≈0\n(两端饱和)',
            xy=(0.15, 0.02), fontsize=9, color=C_GRAY,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
ax.annotate('斜率≈0\n(两端饱和)',
            xy=(0.75, 0.98), fontsize=9, color=C_GRAY,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# 决策边界参考
ax.axhline(y=0.5, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)

ax.set_xlabel('负债率 (DTI)', fontsize=12)
ax.set_ylabel('违约概率', fontsize=12)
ax.set_title('图 3：边际效应在均值处 (MEM)——切线的斜率', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.set_xlim(0, 0.9)
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/027-marginal-effects.png', dpi=200, bbox_inches='tight')
print("✅ 图 3 已保存：027-marginal-effects.png")

# ========== 7. 三种指标对比总结 ==========
print("\n" + "=" * 60)
print("三种指标对比（dti 系数）：")
print("=" * 60)
print(f"  {'指标':<30} {'数值':<20}")
print(f"  {'-'*50}")
print(f"  {'原始系数 (Log-Odds)':<30} {logit_res.params['dti']:<20.3f}")
print(f"  {'几率比 (Odds Ratio)':<30} {odds_ratio['dti']:<20.3f}")
print(f"  {'边际效应 MEM (dy/dx)':<30} {margeff.margeff[0]:<20.3f}")
print()

# ========== 8. 额外：分类评估（混淆矩阵） ==========
df['pred_default'] = (df['logit_fitted'] >= 0.5).astype(int)
tp = ((df['pred_default'] == 1) & (df['default'] == 1)).sum()
tn = ((df['pred_default'] == 0) & (df['default'] == 0)).sum()
fp = ((df['pred_default'] == 1) & (df['default'] == 0)).sum()
fn = ((df['pred_default'] == 0) & (df['default'] == 1)).sum()

print("混淆矩阵（决策阈值 = 0.5）：")
print(f"  {'':>12} {'预测不违约':>10} {'预测违约':>10}")
print(f"  {'实际不违约':>12} {tn:>10} {fp:>10}")
print(f"  {'实际违约':>12} {fn:>10} {tp:>10}")
print(f"\n  准确率: {(tp+tn)/(tp+tn+fp+fn):.2%}")
print(f"  精确率: {tp/(tp+fp):.2%}" if tp+fp > 0 else "  精确率: N/A")
print(f"  召回率: {tp/(tp+fn):.2%}" if tp+fn > 0 else "  召回率: N/A")

print("\n🎉 第 27 篇所有图表与分析已生成完毕！")
