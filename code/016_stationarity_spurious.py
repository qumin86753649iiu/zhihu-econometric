"""
知乎计量经济学系列 — 第 16 篇配套代码
主题：平稳性——伪回归的源头
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

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
print("第 16 篇：平稳性与伪回归")
print("=" * 60)

# ========== 1. 模拟两个独立的随机游走 ==========
print("\n📊 1. 生成两个独立的随机游走...")
n = 200

e1 = np.random.randn(n)
e2 = np.random.randn(n)
X = np.cumsum(e1)        # 随机游走 1
Y = np.cumsum(e2)        # 随机游走 2

# ===== 图 1：两个独立随机游走的可视化 =====
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(X, color=C_MAIN, linewidth=1.8, alpha=0.85, label='随机游走 X')
ax.plot(Y, color=C_RED, linewidth=1.8, alpha=0.85, label='随机游走 Y')
ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xlabel('时间', fontsize=12)
ax.set_ylabel('累积值', fontsize=12)
ax.set_title('图 1：两个完全独立的随机游走——看起来像在"同步"运动？', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/016-spurious-regression.png', dpi=200, bbox_inches='tight')
print("✅ 图 1 已保存：016-spurious-regression.png")

# ========== 2. 伪回归：对两个独立随机游走做 OLS ==========
print("\n🔮 2. 对两个独立随机游走做 OLS 回归（伪回归示范）...")
X_with_const = sm.add_constant(X)
m_spurious = sm.OLS(Y, X_with_const).fit()
print("\n伪回归结果（两个独立随机游走）：")
print(f"  R² = {m_spurious.rsquared:.4f}")
print(f"  adj R² = {m_spurious.rsquared_adj:.4f}")
print(f"  F 统计量 = {m_spurious.fvalue:.2f}, p = {m_spurious.f_pvalue:.6e}")
print(f"  X 的系数 = {m_spurious.params[1]:.4f}")
print(f"  X 的 t 值 = {m_spurious.tvalues[1]:.2f}")
print(f"  X 的 p 值 = {m_spurious.pvalues[1]:.6f}")
if m_spurious.pvalues[1] < 0.05:
    print("  ⚠️ ！！！虽然两个序列完全独立，但回归结果显著——这就是伪回归！")
else:
    print("  本次模拟未得到显著结果（有一定的随机性）")

# ========== 3. 多次模拟验证伪回归的概率 ==========
print("\n🎲 3. 多次模拟验证伪回归的发生概率...")
n_sim = 1000
n_obs = 200
sig_count = 0
results = []

for i in range(n_sim):
    e1_sim = np.random.randn(n_obs)
    e2_sim = np.random.randn(n_obs)
    X_sim = np.cumsum(e1_sim)
    Y_sim = np.cumsum(e2_sim)
    X_sim_c = sm.add_constant(X_sim)
    m_sim = sm.OLS(Y_sim, X_sim_c).fit()
    p_val = m_sim.pvalues[1]
    if p_val < 0.05:
        sig_count += 1

spurious_rate = sig_count / n_sim * 100
print(f"  模拟次数: {n_sim}")
print(f"  每次样本量: {n_obs}")
print(f"  显著（p<0.05）的次数: {sig_count}/{n_sim}")
print(f"  伪回归发生率: {spurious_rate:.1f}%")
print(f"  （理论上应该在 70-80% 左右——Granger & Newbold 1974）")

# ========== 4. ADF 检验 ==========
print("\n🔬 4. ADF 检验——诊断序列的平稳性...")

def adf_report(series, name):
    """对序列做 ADF 检验并输出报告"""
    result = adfuller(series, autolag='AIC')
    print(f"\n  📍 {name}")
    print(f"     ADF 统计量: {result[0]:.4f}")
    print(f"     p 值:       {result[1]:.6f}")
    print(f"     临界值:")
    for key, value in result[4].items():
        print(f"       {key}: {value:.4f}")
    if result[1] < 0.05:
        print(f"     ✅ 结论：p < 0.05，拒绝单位根原假设 → 序列平稳")
    else:
        print(f"     ⚠️ 结论：p ≥ 0.05，不能拒绝单位根原假设 → 序列非平稳")

# 检验原始随机游走
adf_report(X, "随机游走 X（原始）")
adf_report(Y, "随机游走 Y（原始）")

# ===== 图 2：ACF 对比——原始序列 vs 差分序列 =====
X_diff = np.diff(X, n=1)
Y_diff = np.diff(Y, n=1)

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# (a) 原始序列 ACF
plot_acf(X, lags=30, ax=axes[0, 0], title='')
axes[0, 0].set_title('（a）原始非平稳序列 X 的 ACF', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('滞后阶数', fontsize=10)
axes[0, 0].set_ylabel('自相关系数', fontsize=10)

# (b) 差分序列 ACF
plot_acf(X_diff, lags=30, ax=axes[0, 1], title='')
axes[0, 1].set_title('（b）差分后平稳序列 ΔX 的 ACF', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('滞后阶数', fontsize=10)
axes[0, 1].set_ylabel('自相关系数', fontsize=10)

# (c) 原始序列 ACF（Y）
plot_acf(Y, lags=30, ax=axes[1, 0], title='')
axes[1, 0].set_title('（c）原始非平稳序列 Y 的 ACF', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('滞后阶数', fontsize=10)
axes[1, 0].set_ylabel('自相关系数', fontsize=10)

# (d) 差分序列 ACF（Y）
plot_acf(Y_diff, lags=30, ax=axes[1, 1], title='')
axes[1, 1].set_title('（d）差分后平稳序列 ΔY 的 ACF', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('滞后阶数', fontsize=10)
axes[1, 1].set_ylabel('自相关系数', fontsize=10)

plt.suptitle('图 2：ACF 对比——非平稳序列 vs 差分后平稳序列', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/016-adf-demo.png', dpi=200, bbox_inches='tight')
print("\n✅ 图 2 已保存：016-adf-demo.png")

# ADF 检验差分后的序列
adf_report(X_diff, "差分后序列 ΔX")
adf_report(Y_diff, "差分后序列 ΔY")

# ========== 5. 差分前后对比图 ==========
print("\n📈 5. 绘制差分前后对比图...")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# (a) 原始 X
axes[0, 0].plot(X, color=C_MAIN, linewidth=1.5)
axes[0, 0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
axes[0, 0].set_title('（a）原始序列 X（非平稳——均值漂移）', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('时间', fontsize=10)
axes[0, 0].set_ylabel('Xₜ', fontsize=10)
axes[0, 0].grid(alpha=0.2)

# (b) 原始 Y
axes[0, 1].plot(Y, color=C_RED, linewidth=1.5)
axes[0, 1].axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
axes[0, 1].set_title('（b）原始序列 Y（非平稳——均值漂移）', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('时间', fontsize=10)
axes[0, 1].set_ylabel('Yₜ', fontsize=10)
axes[0, 1].grid(alpha=0.2)

# (c) 差分 X
axes[1, 0].plot(X_diff, color=C_GREEN, linewidth=1.5)
axes[1, 0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
axes[1, 0].set_title('（c）差分序列 ΔX（平稳——围绕 0 波动）', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('时间', fontsize=10)
axes[1, 0].set_ylabel('ΔXₜ', fontsize=10)
axes[1, 0].grid(alpha=0.2)

# (d) 差分 Y
axes[1, 1].plot(Y_diff, color=C_PURPLE, linewidth=1.5)
axes[1, 1].axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
axes[1, 1].set_title('（d）差分序列 ΔY（平稳——围绕 0 波动）', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('时间', fontsize=10)
axes[1, 1].set_ylabel('ΔYₜ', fontsize=10)
axes[1, 1].grid(alpha=0.2)

plt.suptitle('图 3：差分前后对比——从非平稳到平稳', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/016-differencing.png', dpi=200, bbox_inches='tight')
print("✅ 图 3 已保存：016-differencing.png")

# ========== 6. 用差分后的序列做回归（正确做法） ==========
print("\n✅ 6. 用差分后的序列（平稳）做回归——正确的做法...")
X_diff_c = sm.add_constant(X_diff)
m_correct = sm.OLS(Y_diff, X_diff_c).fit()
print("\n差分后回归结果（正确的做法）：")
print(f"  R² = {m_correct.rsquared:.4f}")
print(f"  adj R² = {m_correct.rsquared_adj:.4f}")
print(f"  X_diff 的 p 值 = {m_correct.pvalues[1]:.6f}")
if m_correct.pvalues[1] >= 0.05:
    print("  ✅ 正确：差分后两个序列不再'显著相关'——这才真实反映了它们之间没有关系")
else:
    print("  ⚠️ 注意：即使差分后也可能偶然显著（5% 的假阳性率）")

# ========== 7. 总结输出 ==========
print("\n" + "=" * 60)
print("📋 总结")
print("=" * 60)
print(f"""
伪回归模拟:
  - 两个独立随机游走回归: R² = {m_spurious.rsquared:.3f}, X 的 p = {m_spurious.pvalues[1]:.6f}
  - 伪回归发生率（{n_sim} 次模拟）: {spurious_rate:.1f}%

ADF 检验（原始序列）:
  - X: p = {adfuller(X, autolag='AIC')[1]:.6f}  → {"平稳 ✅" if adfuller(X, autolag='AIC')[1] < 0.05 else "非平稳 ⚠️"}
  - Y: p = {adfuller(Y, autolag='AIC')[1]:.6f}  → {"平稳 ✅" if adfuller(Y, autolag='AIC')[1] < 0.05 else "非平稳 ⚠️"}

ADF 检验（差分后）:
  - ΔX: p = {adfuller(X_diff, autolag='AIC')[1]:.6f}  → {"平稳 ✅" if adfuller(X_diff, autolag='AIC')[1] < 0.05 else "非平稳 ⚠️"}
  - ΔY: p = {adfuller(Y_diff, autolag='AIC')[1]:.6f}  → {"平稳 ✅" if adfuller(Y_diff, autolag='AIC')[1] < 0.05 else "非平稳 ⚠️"}

差分后回归:
  - R² = {m_correct.rsquared:.3f}, p = {m_correct.pvalues[1]:.6f}
  - {"不存在伪回归 ✅" if m_correct.pvalues[1] >= 0.05 else "偶然显著（假阳性）"}
""")
print("🎉 第 16 篇所有图表已生成完毕！")
