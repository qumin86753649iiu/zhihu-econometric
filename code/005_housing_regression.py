"""
知乎计量经济学系列 — 第 5 篇配套代码
主题：完整的线性回归实战 — 用房价数据串联前 4 篇
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
C_MAIN = '#2E86AB'
C_ACCENT = '#F18F01'
C_RED = '#C73E1D'
C_GREEN = '#4C9F70'

# ========== 1. 生成模拟房价数据 ==========
np.random.seed(42)
n = 200

# 解释变量
area = np.random.uniform(40, 180, n)           # 面积 40-180 平米
bedrooms = np.random.randint(1, 6, n)           # 卧室数 1-5
floor = np.random.randint(1, 35, n)             # 楼层
location = np.random.uniform(1, 10, n)          # 位置评分 1-10
age = np.random.uniform(0, 30, n)               # 房龄 0-30 年

# 真实关系：面积最重要，位置其次，房龄负向
price = (30 + 2.8 * area + 5 * location
         - 1.5 * age + 8 * bedrooms
         + np.random.normal(0, 40, n))
price = np.maximum(price, 30)  # 房价不低于30万

df = pd.DataFrame({
    'price': price,
    'area': area,
    'bedrooms': bedrooms,
    'floor': floor,
    'location': location,
    'age': age
})

print("=== 数据概览 ===")
print(f"样本量: {n}")
print(df.describe().round(1))

# ========== 2. 单变量回归：面积 → 价格（第 1-2 篇的延伸）==========
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.scatter(df['area'], df['price'], alpha=0.5, color=C_MAIN, edgecolors='white', linewidth=0.5, s=40)

X1 = sm.add_constant(df['area'])
m1 = sm.OLS(df['price'], X1).fit()
xp = np.linspace(30, 190, 100)
yp = m1.params.iloc[0] + m1.params.iloc[1] * xp
ax.plot(xp, yp, color=C_RED, linewidth=2.5,
        label=f'价格 = {m1.params.iloc[0]:.0f} + {m1.params.iloc[1]:.2f} x 面积\n(R² = {m1.rsquared:.2f})')

# 标注置信区间
pred = m1.get_prediction(sm.add_constant(xp))
pred_ci = pred.summary_frame(alpha=0.05)
ax.fill_between(xp, pred_ci['mean_ci_lower'], pred_ci['mean_ci_upper'],
                color=C_RED, alpha=0.08)

ax.set_xlabel('面积（平方米）', fontsize=12)
ax.set_ylabel('总价（万元）', fontsize=12)
ax.set_title('图1：面积 vs 房价 — 第 1-2 篇的知识点', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\005-price-area.png',
            dpi=200, bbox_inches='tight')
print("\n✅ 面积-房价图已保存")

# ========== 3. 多元回归：一起上（第 6 篇的预告，但这里展示用）==========
X_multi = sm.add_constant(df[['area', 'location', 'age', 'bedrooms', 'floor']])
m_multi = sm.OLS(df['price'], X_multi).fit()

print("\n=== 多元回归结果 ===")
print(m_multi.summary())

# 系数可视化
fig, ax = plt.subplots(figsize=(9, 5))
coefs = m_multi.params.iloc[1:]  # 去掉截距
errors = m_multi.bse.iloc[1:]
var_names = ['面积', '位置评分', '房龄', '卧室数', '楼层']
colors = [C_MAIN, C_GREEN, C_RED, C_ACCENT, '#A23B72']

ax.barh(range(len(coefs)), coefs, xerr=1.96*errors, capsize=3,
        color=colors, alpha=0.75, edgecolor='white')
ax.set_yticks(range(len(coefs)))
ax.set_yticklabels(var_names)
ax.axvline(x=0, color='gray', linewidth=1, linestyle='-')
ax.set_xlabel('系数（边际效应：每单位变化对房价的影响，万元）', fontsize=11)
ax.set_title('图2：多元回归中各变量的"净效应"', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.2)
# 加显著性标注
for i, (c, e, p) in enumerate(zip(coefs, errors, m_multi.pvalues.iloc[1:])):
    stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    ax.text(c + 0.3 + 1.96*e if c > 0 else c - 0.3 - 1.96*e, i,
            f'{c:.1f}{stars}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\005-multi-coefficients.png',
            dpi=200, bbox_inches='tight')
print("✅ 系数对比图已保存")

# ========== 4. 遗漏变量对比（第 3 篇的复习）===========
# 只放面积 → 有偏 vs 全部变量 → 无偏
m_biased = sm.OLS(df['price'], sm.add_constant(df['area'])).fit()
beta_biased = m_biased.params.iloc[1]
beta_unbiased = m_multi.params['area']

print(f"""
╔═══════════════════════════════════════════════╗
║  遗漏变量对比                                 ║
╠═══════════════════════════════════════════════╣
║  只有面积:   面积每+1㎡, 价格+{beta_biased:.1f}万      ║
║  控制全部:   面积每+1㎡, 价格+{beta_unbiased:.1f}万      ║
║  真实值:     2.8 万/㎡                       ║
╚═══════════════════════════════════════════════╝
""")

# ========== 5. 残差诊断（第 4 篇的复习：信噪比）============
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左：拟合值 vs 残差
ax1 = axes[0]
ax1.scatter(m_multi.fittedvalues, m_multi.resid, alpha=0.5, color=C_MAIN, s=30)
ax1.axhline(y=0, color='gray', linewidth=1, linestyle='--')
ax1.set_xlabel('拟合值（预测价格，万元）', fontsize=10)
ax1.set_ylabel('残差（实际 - 预测，万元）', fontsize=10)
ax1.set_title(f'图3a：残差 vs 拟合值（看异方差）', fontsize=11, fontweight='bold')
ax1.grid(alpha=0.2)

# 右：Q-Q 图
ax2 = axes[1]
from scipy import stats
stats.probplot(m_multi.resid, dist="norm", plot=ax2)
ax2.get_lines()[0].set_markerfacecolor(C_MAIN)
ax2.get_lines()[0].set_markeredgecolor('white')
ax2.get_lines()[0].set_markersize(4)
ax2.get_lines()[1].set_color(C_RED)
ax2.get_lines()[1].set_linewidth(2)
ax2.set_title('图3b：Q-Q 图（看残差是否正态）', fontsize=11, fontweight='bold')
ax2.grid(alpha=0.2)

plt.suptitle('图3：回归诊断 — 检查模型假设是否成立', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\005-diagnostics.png',
            dpi=200, bbox_inches='tight')
print("✅ 诊断图已保存")

# ========== 6. 结果摘要（第 4 篇的 p 值、t 值解读）============
print(f"""
╔════════════════════════════════════════════════════╗
║  完整回归结果解读（多元模型）                         ║
╠════════════════════════════════════════════════════╣
║  R² = {m_multi.rsquared:.3f}                        ║
║  调整 R² = {m_multi.rsquared_adj:.3f}               ║
║  F = {m_multi.fvalue:.1f} (p = {m_multi.f_pvalue:.2e})  ║
╠════════════════════════════════════════════════════╣
║  变量     系数    标准误   t值    p值   显著?        ║
║  ────────────────────────────────────────────── ║
""")

for name, coef, se, t, p in zip(
    ['面积', '位置', '房龄', '卧室', '楼层'],
    m_multi.params.iloc[1:], m_multi.bse.iloc[1:],
    m_multi.tvalues.iloc[1:], m_multi.pvalues.iloc[1:]):
    stars = '***' if p < 0.01 else '**' if p < 0.05 else '*'
    print(f"  {name:>4} {coef:>8.1f} {se:>8.2f} {t:>6.2f} {p:>8.4f} {stars:>4}")

print("  ──────────────────────────────────────────────")
print(f"""  R² = {m_multi.rsquared:.3f}, 调整 R² = {m_multi.rsquared_adj:.3f}
╚════════════════════════════════════════════════════╝
""")
