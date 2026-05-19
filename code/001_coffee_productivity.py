"""
知乎计量经济学系列 — 第 1 篇配套代码
主题：咖啡摄入量与代码行数的关系
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ========== 统一配置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
COLOR_MAIN = '#2E86AB'
COLOR_ACCENT = '#F18F01'
COLOR_SCATTER = '#2E86AB'

# ========== 1. 生成模拟数据 ==========
np.random.seed(42)
n = 50
coffee = np.random.uniform(0, 5, n)
code_lines = 150 + 80 * coffee + np.random.normal(0, 100, n)

df = pd.DataFrame({
    'coffee_cups': coffee,
    'code_lines': code_lines
})

print("=== 数据前5行 ===")
print(df.head())
print(f"\n样本量: {n}")
print(f"咖啡摄入范围: {coffee.min():.1f} ~ {coffee.max():.1f} 杯")
print(f"代码行数范围: {code_lines.min():.0f} ~ {code_lines.max():.0f} 行")

# ========== 2. 散点图 ==========
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.scatter(df['coffee_cups'], df['code_lines'], alpha=0.65,
           color=COLOR_SCATTER, edgecolors='white', linewidth=0.5, s=60)
ax.set_xlabel('咖啡摄入量（杯）', fontsize=12)
ax.set_ylabel('当天写代码行数', fontsize=12)
ax.set_title('图1：咖啡 vs 代码行数 — 看起来确实有关系', fontsize=13, fontweight='bold')
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\001-scatter.png',
            dpi=200, bbox_inches='tight')
print("\n✅ 散点图已保存: images/001-scatter.png")

# ========== 3. 回归拟合 — 带回归线和置信带 ==========
X = sm.add_constant(df['coffee_cups'])
model = sm.OLS(df['code_lines'], X).fit()
print(model.summary())

# 预测 + 置信区间
x_pred = np.linspace(0, 5.5, 100)
X_pred = sm.add_constant(x_pred)
pred = model.get_prediction(X_pred)
pred_summary = pred.summary_frame(alpha=0.05)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.scatter(df['coffee_cups'], df['code_lines'], alpha=0.5,
           color=COLOR_SCATTER, edgecolors='white', linewidth=0.5, s=50)
ax.plot(x_pred, pred_summary['mean'], color='#C73E1D', linewidth=2.5,
        label='回归线 (β₁ = 78.6)')
ax.fill_between(x_pred,
                pred_summary['mean_ci_lower'],
                pred_summary['mean_ci_upper'],
                color='#C73E1D', alpha=0.12,
                label='95% 置信区间')
ax.set_xlabel('咖啡摄入量（杯）', fontsize=12)
ax.set_ylabel('当天写代码行数', fontsize=12)
ax.set_title('图2：拟合回归线 — 每多一杯咖啡，多写 75 行代码', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\001-regression.png',
            dpi=200, bbox_inches='tight')
print("✅ 回归图已保存: images/001-regression.png")

# ========== 4. 结果摘要 ==========
beta1 = model.params['coffee_cups']
pval = model.pvalues['coffee_cups']
r2 = model.rsquared
ci_low, ci_high = model.conf_int().loc['coffee_cups']
print(f"""
╔══════════════════════════════════════════╗
║  回归结果摘要                            ║
╠══════════════════════════════════════════╣
║  系数 β₁ (咖啡) = {beta1:>8.2f}              ║
║  P 值           = {pval:>8.4f}              ║
║  R²            = {r2:>8.3f}              ║
║  95% CI        = [{ci_low:>7.1f}, {ci_high:>7.1f}]      ║
╚══════════════════════════════════════════╝
""")
