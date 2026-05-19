"""
知乎计量经济学系列 — 第 19 篇配套代码
主题：双重差分——修地铁到底有没有让房价涨？
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
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
print("第 19 篇：双重差分——修地铁到底有没有让房价涨？")
print("=" * 55)

# ========== 1. 模拟面板数据 ==========
print("\n>>> 正在模拟面板数据...")

n_months = 24
months = np.arange(1, n_months + 1)
treatment_month = 13  # 第 13 个月开始修地铁

# 真实参数
base_a = 100      # A 市（处理组）基期房价
base_b = 95       # B 市（对照组）基期房价
trend = 0.5       # 共同的房价上涨趋势（平行趋势假设）
treatment_effect = 8   # 地铁的真实因果效应

# 生成房价数据
price_a = np.zeros(n_months)
price_b = np.zeros(n_months)

for t in range(n_months):
    # A 市（修地铁）
    noise_a = np.random.normal(0, 2)
    price_a[t] = base_a + trend * months[t] + noise_a
    if months[t] >= treatment_month:
        price_a[t] += treatment_effect

    # B 市（没修地铁）
    noise_b = np.random.normal(0, 2)
    price_b[t] = base_b + trend * months[t] + noise_b

# 合并为面板数据
df = pd.DataFrame({
    'month': np.tile(months, 2),
    'price': np.concatenate([price_a, price_b]),
    'treat': np.array([1] * n_months + [0] * n_months, dtype=int),
    'period': np.array(
        [1 if m >= treatment_month else 0 for m in months] * 2,
        dtype=int
    )
})
df['treatXperiod'] = df['treat'] * df['period']

# 汇总统计
print(f"   生成了 {n_months} 个月的面板数据（{len(df)} 条观测）")
print(f"   处理组（A 市）: 前 {treatment_month - 1} 个月 + 后 {n_months - treatment_month + 1} 个月")
print(f"   对照组（B 市）: 全程 {n_months} 个月")
print(f"\n   真实因果效应（地铁对房价的影响）: {treatment_effect} 单位\n")

# 分组均值
print("各组均值：")
print(df.groupby(['treat', 'period'])['price'].describe())

# ========== 图 1：平行趋势 + 政策冲击图（事件图）==========
print("\n>>> 正在绘制 DID 事件图...")

fig, ax = plt.subplots(figsize=(14, 7))

# 两条时间线
ax.plot(months, price_a, color=C_MAIN, linewidth=2.5, marker='o',
        markersize=5, label='A 市（修地铁）', zorder=3)
ax.plot(months, price_b, color=C_ACCENT, linewidth=2.5, marker='s',
        markersize=5, label='B 市（未修地铁）', zorder=3)

# 标出处理期
ax.axvspan(treatment_month - 0.5, n_months + 0.5, color=C_RED, alpha=0.06, zorder=1)
ax.axvline(x=treatment_month - 0.5, color=C_RED, linewidth=1.5, linestyle='--',
           alpha=0.6, label='地铁开通', zorder=2)

# 标注"处理前"和"处理后"
ylim = ax.get_ylim()
ax.text(treatment_month / 2 - 0.5, ylim[0] + 3, '处 理 前',
        ha='center', fontsize=16, fontweight='bold', color=C_GRAY, alpha=0.4)
ax.text(treatment_month + (n_months - treatment_month) / 2,
        ylim[0] + 3, '处 理 后',
        ha='center', fontsize=16, fontweight='bold', color=C_RED, alpha=0.4)

# 标注 DID 效应
y_a_before = price_a[:treatment_month - 1].mean()
y_a_after = price_a[treatment_month - 1:].mean()
y_b_before = price_b[:treatment_month - 1].mean()
y_b_after = price_b[treatment_month - 1:].mean()

did_estimate = (y_a_after - y_a_before) - (y_b_after - y_b_before)
ax.annotate(
    f'DID 估计 = {did_estimate:.1f}',
    xy=(treatment_month + 2, y_a_after),
    xytext=(treatment_month + 5, y_a_after + 4),
    fontsize=14, fontweight='bold', color=C_RED,
    arrowprops=dict(facecolor=C_RED, arrowstyle='->', lw=2),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85)
)

ax.set_xlabel('月份', fontsize=13)
ax.set_ylabel('平均房价（万元/平）', fontsize=13)
ax.set_title('图 1：修地铁对房价的影响——双重差分直觉', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.set_xlim(0.5, n_months + 0.5)
ax.set_xticks(np.arange(1, n_months + 1, 2))
ax.grid(alpha=0.15)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/019-did-parallel-plot.png', dpi=200, bbox_inches='tight')
print("✅ 图 1 已保存：DID 事件图")

# ========== 图 2：DID 估计量的四柱图 ==========
print("\n>>> 正在绘制 DID 估计量分解图...")

fig, ax = plt.subplots(figsize=(10, 7))

# 四组均值
groups = ['A 市\n处理前', 'A 市\n处理后', 'B 市\n处理前', 'B 市\n处理后']
values = [y_a_before, y_a_after, y_b_before, y_b_after]
colors_bar = [C_LIGHT_BLUE, C_MAIN, '#FDE2B5', C_ACCENT]

bars = ax.bar(groups, values, color=colors_bar, width=0.6, edgecolor='white', linewidth=1.5)

# 在柱子上标注数值
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 画箭头展示 DID: A_after - A_before
diff_a = y_a_after - y_a_before
mid_a = (y_a_before + y_a_after) / 2
ax.annotate(
    f'ΔA = {diff_a:.1f}',
    xy=(0.2, mid_a),
    xytext=(0.55, mid_a),
    fontsize=11, color=C_RED, fontweight='bold',
    arrowprops=dict(arrowstyle='<->', color=C_RED, lw=2)
)

# 画箭头展示 DID: B_after - B_before
diff_b = y_b_after - y_b_before
mid_b = (y_b_before + y_b_after) / 2
ax.annotate(
    f'ΔB = {diff_b:.1f}',
    xy=(2.2, mid_b),
    xytext=(2.55, mid_b),
    fontsize=11, color=C_RED, fontweight='bold',
    arrowprops=dict(arrowstyle='<->', color=C_RED, lw=2)
)

# DID = ΔA - ΔB
ax.text(1.5, min(values) - 5,
        f'DID = ΔA − ΔB = {diff_a:.1f} − {diff_b:.1f} = {did_estimate:.1f}',
        ha='center', fontsize=14, fontweight='bold',
        color=C_RED,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3E0', alpha=0.9))

ax.set_ylabel('平均房价（万元/平）', fontsize=12)
ax.set_title('图 2：DID 估计量的分解——"差异的差异"', fontsize=14, fontweight='bold')
ax.grid(alpha=0.1, axis='y')

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/019-did-estimation.png', dpi=200, bbox_inches='tight')
print("✅ 图 2 已保存：DID 估计量分解图")

# ========== 3. DID 回归：交互项 ==========
print("\n>>> 正在运行 DID 回归...")

# OLS: price = β₀ + β₁·treat + β₂·post + β₃·treat×post + ε
model = sm.OLS(
    df['price'],
    sm.add_constant(df[['treat', 'period', 'treatXperiod']])
).fit()

print("\n" + "=" * 55)
print("DID 回归结果")
print("=" * 55)
print(model.summary())

# 提取系数
coefs = model.params
cis = model.conf_int()

print(f"\n核心解读：")
print(f"  β₀ (const)         = {coefs['const']:.3f}  → B 市处理前的基准房价")
print(f"  β₁ (treat)         = {coefs['treat']:.3f}  → A 市与 B 市的固定差异（基期）")
print(f"  β₂ (period)        = {coefs['period']:.3f}  → B 市处理前后的时间趋势")
print(f"  β₃ (treatXperiod)  = {coefs['treatXperiod']:.3f}  → **DID 估计量（因果效应）**")

# 对比真实效应
print(f"\n  真实因果效应 = {treatment_effect}")
print(f"  DID 估计量   = {coefs['treatXperiod']:.3f}")
print(f"  估计误差     = {coefs['treatXperiod'] - treatment_effect:.3f}")

# ===== 图 3：回归系数图 =====
print("\n>>> 正在绘制回归系数图...")

fig, ax = plt.subplots(figsize=(10, 6))

coef_names = ['const\n(截距)', 'treat\n(A 市 vs B 市)',
              'period\n(时间趋势)', 'treat×period\n(DID 效应)']
coef_vals = [coefs['const'], coefs['treat'], coefs['period'], coefs['treatXperiod']]
ci_lower = [cis[0]['const'], cis[0]['treat'], cis[0]['period'], cis[0]['treatXperiod']]
ci_upper = [cis[1]['const'], cis[1]['treat'], cis[1]['period'], cis[1]['treatXperiod']]
bar_colors = [C_GRAY, C_PURPLE, C_GREEN, C_RED]

# 显著性星标
pvalues = model.pvalues
sig_stars = [
    '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    for p in [pvalues['const'], pvalues['treat'],
              pvalues['period'], pvalues['treatXperiod']]
]

y_pos = np.arange(len(coef_names))
# 构造不对称的 error bar
err_low = np.array([coef_vals[i] - ci_lower[i] for i in range(len(coef_vals))])
err_high = np.array([ci_upper[i] - coef_vals[i] for i in range(len(coef_vals))])

ax.barh(y_pos, coef_vals,
        xerr=(err_low, err_high),
        color=bar_colors, edgecolor='white', linewidth=1.5,
        capsize=5, height=0.6)

# 标注系数值和显著性
for i, (v, star) in enumerate(zip(coef_vals, sig_stars)):
    offset = 0.3 if v >= 0 else -0.3
    ax.text(v + offset, i, f'{v:.3f} {star}',
            va='center', ha='left' if v >= 0 else 'right',
            fontsize=11, fontweight='bold')

ax.axvline(x=0, color='black', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(coef_names, fontsize=11)
ax.set_xlabel('系数估计值', fontsize=12)
ax.set_title('图 3：DID 回归系数——交互项就是因果效应', fontsize=14, fontweight='bold')
ax.grid(alpha=0.15, axis='x')

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/019-did-regression.png', dpi=200, bbox_inches='tight')
print("✅ 图 3 已保存：回归系数图")

# ========== 4. 额外分析：用公式形式做 DID ==========
print("\n>>> 用公式形式复现 DID 回归...")

model_formula = smf.ols('price ~ treat * period', data=df).fit()
print(model_formula.summary())

# ========== 5. DID 的本质 ==========
print("\n" + "=" * 55)
print("手动计算 DID 估计量（验证回归结果）：")
print("=" * 55)

print(f"""
  A 市处理前均值:   {y_a_before:.3f}
  A 市处理后均值:   {y_a_after:.3f}
  ΔA = 处理后 − 处理前 = {y_a_after - y_a_before:.3f}

  B 市处理前均值:   {y_b_before:.3f}
  B 市处理后均值:   {y_b_after:.3f}
  ΔB = 处理后 − 处理前 = {y_b_after - y_b_before:.3f}

  DID = ΔA − ΔB = {(y_a_after - y_a_before) - (y_b_after - y_b_before):.3f}
  ──────────────────────────────────────
  回归交互项系数（treat×period）: {coefs['treatXperiod']:.3f}
  完全一致 ✅
""")

print("\n" + "=" * 55)
print("🎉 第 19 篇全部图表已生成完毕！")
print("=" * 55)
