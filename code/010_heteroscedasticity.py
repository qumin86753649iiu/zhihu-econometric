"""
知乎计量经济学系列 — 第 10 篇配套代码
主题：异方差——误差的方差不再是常数
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

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

# ========== 1. 模拟数据：含异方差的收入-消费关系 ==========
np.random.seed(42)
n = 300
income = np.random.uniform(3, 50, n)  # 月收入（千元）

# 异方差：消费的波动随收入增加
consumption = (
    3 + 0.6 * income
    + np.random.normal(0, 1, n) * (1 + 0.3 * income)
)

df = pd.DataFrame({'income': income, 'consumption': consumption})

# ===== 图 1：收入 vs 消费（散点图+回归线，突出异方差） =====
X = sm.add_constant(df['income'])
m = sm.OLS(df['consumption'], X).fit()
df['resid'] = m.resid
df['fitted'] = m.fittedvalues

fig, ax = plt.subplots(figsize=(10, 6))

# 散点
ax.scatter(df['income'], df['consumption'], alpha=0.5,
           color=C_MAIN, edgecolors='white', linewidth=0.3, s=35)

# 回归线
xp = np.linspace(0, 55, 200)
yp = m.params.iloc[0] + m.params.iloc[1] * xp
ax.plot(xp, yp, color=C_RED, linewidth=2.5, label=f'消费 = {m.params.iloc[0]:.2f} + {m.params.iloc[1]:.2f} × 收入')

# 添加异方差注释——用两条竖线区间示意
# 低收入的波动范围
low_band_x = 10
low_band_y = m.params.iloc[0] + m.params.iloc[1] * low_band_x
low_resid_subset = df.loc[np.abs(df['income'] - low_band_x) < 2, 'resid']
if len(low_resid_subset) > 0:
    low_std = low_resid_subset.std()
    ax.annotate('低收入人群\n波动小', xy=(low_band_x, low_band_y + 3 * low_std),
                xytext=(low_band_x - 8, low_band_y + 12),
                fontsize=9, color='#4C9F70', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=1.5))

# 高收入的波动范围
high_band_x = 40
high_band_y = m.params.iloc[0] + m.params.iloc[1] * high_band_x
high_resid_subset = df.loc[np.abs(df['income'] - high_band_x) < 2, 'resid']
if len(high_resid_subset) > 0:
    high_std = high_resid_subset.std()
    ax.annotate('高收入人群\n波动大', xy=(high_band_x, high_band_y + 3 * high_std),
                xytext=(high_band_x + 2, high_band_y + 18),
                fontsize=9, color=C_RED, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.5))

ax.set_xlabel('月收入（千元）', fontsize=12)
ax.set_ylabel('月消费（千元）', fontsize=12)
ax.set_title('图 1：收入 vs 消费——方差随收入增大', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.2)
ax.set_xlim(0, 55)
ax.set_ylim(0, 55)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/010-income-consumption-scatter.png',
            dpi=200, bbox_inches='tight')
print("✅ 图 1 已保存：收入-消费散点图")

# ========== 2. 图 2：残差诊断 ==========
df['resid'] = m.resid
df['fitted'] = m.fittedvalues

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# 左：残差 vs 拟合值
ax1 = axes[0]
ax1.scatter(df['fitted'], df['resid'], alpha=0.4, color=C_MAIN,
            edgecolors='white', linewidth=0.2, s=30)
ax1.axhline(y=0, color='gray', linewidth=1, linestyle='--')
# 添加"喇叭形"的轮廓示意
x_range = np.linspace(df['fitted'].min(), df['fitted'].max(), 100)
# 拟合上下界展示异方差
from numpy.polynomial import polynomial as P
abs_resid = np.abs(df['resid'])
# 用二次多项式拟合残差绝对值，展示喇叭形趋势
coeffs = P.polyfit(df['fitted'], abs_resid, 2)
trend = P.polyval(x_range, coeffs)
ax1.plot(x_range, trend, color=C_RED, linewidth=2, linestyle='--',
         alpha=0.8, label='残差绝对值的趋势')
ax1.plot(x_range, -trend, color=C_RED, linewidth=2, linestyle='--', alpha=0.8)
ax1.fill_between(x_range, -trend, trend, color=C_RED, alpha=0.06)
ax1.set_xlabel('拟合值（预测消费，千元）', fontsize=11)
ax1.set_ylabel('残差（实际 - 预测，千元）', fontsize=11)
ax1.set_title('（a）残差 vs 拟合值', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.2)
ax1.legend(fontsize=9)

# 右：残差 vs 收入
ax2 = axes[1]
ax2.scatter(df['income'], df['resid'], alpha=0.4, color=C_GREEN,
            edgecolors='white', linewidth=0.2, s=30)
ax2.axhline(y=0, color='gray', linewidth=1, linestyle='--')
# 同样拟合喇叭形趋势
coeffs2 = P.polyfit(df['income'], abs_resid, 2)
trend2 = P.polyval(x_range, coeffs2)
ax2.plot(x_range, trend2, color=C_RED, linewidth=2, linestyle='--',
         alpha=0.8, label='残差绝对值的趋势')
ax2.plot(x_range, -trend2, color=C_RED, linewidth=2, linestyle='--', alpha=0.8)
ax2.fill_between(x_range, -trend2, trend2, color=C_RED, alpha=0.06)
ax2.set_xlabel('月收入（千元）', fontsize=11)
ax2.set_ylabel('残差（实际 - 预测，千元）', fontsize=11)
ax2.set_title('（b）残差 vs 收入', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.2)
ax2.legend(fontsize=9)

plt.suptitle('图 2：残差诊断——清晰可见的"喇叭形"', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/010-residual-diagnostics.png',
            dpi=200, bbox_inches='tight')
print("✅ 图 2 已保存：残差诊断图")

# ========== 3. Breusch-Pagan 检验输出 ==========
bp_test = het_breuschpagan(m.resid, X)
print(f"\n{'='*50}")
print("Breusch-Pagan 检验结果：")
print(f"  LM 统计量: {bp_test[0]:.2f}")
print(f"  LM p值:    {bp_test[1]:.6f}")
print(f"  F 统计量:  {bp_test[2]:.2f}")
print(f"  F p值:     {bp_test[3]:.6f}")
if bp_test[1] < 0.05:
    print("  ✅ 结论：拒绝同方差原假设 → 存在异方差")
else:
    print("  ❌ 结论：不能拒绝同方差原假设")
print(f"{'='*50}\n")

# ========== 4. 稳健标准误对比 ==========
m_robust_hc3 = m.get_robustcov_results(cov_type='HC3')

# 构造对比表
se_ols = m.bse
se_robust = m_robust_hc3.bse
coefs = m.params

print(f"{'变量':>8} {'系数':>8} {'普通SE':>8} {'稳健SE':>8} {'变化%':>8}")
print("-" * 48)
for name, coef, se_o, se_r in zip(['const', 'income'], coefs, se_ols, se_robust):
    change = (se_r - se_o) / se_o * 100
    print(f"{name:>8} {coef:>8.3f} {se_o:>8.3f} {se_r:>8.3f} {change:>+7.1f}%")
print("-" * 48)

# ===== 图 3：标准误对比条形图 =====
fig, ax = plt.subplots(figsize=(8, 5))

labels = ['截距 (const)', '收入 (income)']
x = np.arange(len(labels))
width = 0.35

bars1 = ax.bar(x - width/2, se_ols, width, label='普通标准误',
               color=C_GRAY, alpha=0.7, edgecolor='white')
bars2 = ax.bar(x + width/2, se_robust, width, label='稳健标准误 (HC3)',
               color=C_ACCENT, alpha=0.8, edgecolor='white')

# 标注变化百分比
for i, (se_o, se_r) in enumerate(zip(se_ols, se_robust)):
    change = (se_r - se_o) / se_o * 100
    sign = '+' if change > 0 else ''
    ax.text(i + width/2, se_r + 0.02, f'{sign}{change:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            color=C_RED if change > 0 else C_GREEN)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('标准误', fontsize=12)
ax.set_title('图 3：普通标准误 vs 稳健标准误——稳健标准误更「诚实」', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.2)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/010-standard-errors-comparison.png',
            dpi=200, bbox_inches='tight')
print("✅ 图 3 已保存：标准误对比图")

# ========== 5. 回归报告函数示例 ==========
def regression_report(m_ols, cov_type='HC3'):
    """输出带稳健标准误的回归结果"""
    m_robust = m_ols.get_robustcov_results(cov_type=cov_type)
    bp = het_breuschpagan(m_ols.resid, m_ols.model.exog)
    has_hete = "⚠️ 存在异方差" if bp[1] < 0.05 else "✅ 同方差（基本）"

    print(f"""
╔═══════════════════════════════════════╗
║  回归报告 + 异方差诊断                  ║
╠═══════════════════════════════════════╣
║  模型: OLS  (稳健标准误: {cov_type})    ║
║  BP检验: LM={bp[0]:.2f}, p={bp[1]:.4f}   ║
║  诊断:   {has_hete}         ║
║  R² = {m_ols.rsquared:.3f}                     ║
╠═══════════════════════════════════════╣
""")
    print(m_robust.summary())

print("\n" + "="*50)
print("回归报告示例（带稳健标准误）：")
print("="*50)
regression_report(m)

print("\n🎉 第 10 篇所有图表已生成完毕！")
