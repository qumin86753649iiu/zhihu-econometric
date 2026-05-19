"""
知乎计量经济学系列 — 第 9 篇配套代码
主题：交互效应 — 咖啡×睡眠，效应不是简单相加
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
C_MAIN = '#2E86AB'
C_ACCENT = '#F18F01'
C_RED = '#C73E1D'
C_GREEN = '#4C9F70'

# ========== 1. 生成数据：咖啡 × 睡眠 ==========
np.random.seed(42)
n = 150

coffee = np.random.uniform(0, 5, n)       # 0-5 杯咖啡
sleep = np.random.uniform(4, 10, n)       # 4-10 小时睡眠
sleep_binary = (sleep >= 7).astype(int)   # 是否睡够 7 小时

# 真实效应：
# - 咖啡提神，但仅当睡够了才有效（交互！）
# - 没睡够的人，咖啡只增加焦虑不减疲劳
# - 睡够的人，咖啡大幅提升效率
productivity = (50 + 10 * coffee + 5 * sleep
                + 15 * coffee * (sleep - 7)  # 交互项：每多睡1小时，咖啡的效果增加15
                + np.random.normal(0, 20, n))
productivity = np.maximum(productivity, 10)

df = pd.DataFrame({
    'productivity': productivity,
    'coffee': coffee,
    'sleep': sleep,
    'sleep_enough': sleep_binary
})

print("=== 数据前5行 ===")
print(df.head())

# ========== 2. 不加交互项的"错误"模型 ==========
X_simple = sm.add_constant(df[['coffee', 'sleep']])
m_simple = sm.OLS(df['productivity'], X_simple).fit()

# ========== 3. 加交互项的正确模型 ==========
# 方式 1：连续变量交互 sleep × coffee
df['coffee_x_sleep'] = df['coffee'] * (df['sleep'] - 7)  # 中心化避免共线性
X_interact = sm.add_constant(df[['coffee', 'sleep', 'coffee_x_sleep']])
m_interact = sm.OLS(df['productivity'], X_interact).fit()

# 方式 2：分类交互 sleep_enough × coffee
df['coffee_x_enough'] = df['coffee'] * df['sleep_enough']
X_binary = sm.add_constant(df[['coffee', 'sleep_enough', 'coffee_x_enough']])
m_binary = sm.OLS(df['productivity'], X_binary).fit()

print("\n=== 不加交互项（错误的模型）===")
print(m_simple.summary())
print("\n=== 加交互项（正确的模型）===")
print(m_interact.summary())

# ========== 4. 可视化一：按"是否睡够"分组画回归线 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：分组散点 + 分组回归线
ax1 = axes[0]
colors_grp = {0: C_ACCENT, 1: C_MAIN}
labels_grp = {0: '没睡够 (<7h)', 1: '睡够了 (≥7h)'}

for enough in [0, 1]:
    mask = df['sleep_enough'] == enough
    subset = df[mask]
    ax1.scatter(subset['coffee'], subset['productivity'],
                alpha=0.4, color=colors_grp[enough], s=30,
                label=labels_grp[enough])
    # 分组回归
    X_grp = sm.add_constant(subset['coffee'])
    m_grp = sm.OLS(subset['productivity'], X_grp).fit()
    xp = np.linspace(0, 5, 50)
    yp = m_grp.params.iloc[0] + m_grp.params.iloc[1] * xp
    label_line = f'{labels_grp[enough]}: β={m_grp.params.iloc[1]:.0f}'
    ax1.plot(xp, yp, color=colors_grp[enough], linewidth=2.5, label=label_line)

ax1.set_xlabel('咖啡（杯/天）', fontsize=11)
ax1.set_ylabel('工作效率指数', fontsize=11)
ax1.set_title('图1a：分组建模 — 咖啡对效率的影响取决于睡眠', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.2)

# 右图：手动展示交互项的含义
ax2 = axes[1]
coffee_range = np.linspace(0, 5, 20)

for sleep_val, color, label in [
    (5, C_ACCENT, '只睡 5h'),
    (7, C_GREEN, '睡 7h'),
    (9, C_MAIN, '睡 9h')]:
    
    y_pred = (m_interact.params.iloc[0] + m_interact.params.iloc[2] * sleep_val
              + (m_interact.params.iloc[1] + m_interact.params.iloc[3] * (sleep_val - 7)) * coffee_range)
    ax2.plot(coffee_range, y_pred, color=color, linewidth=2.5, label=label)
    slope = m_interact.params.iloc[1] + m_interact.params.iloc[3] * (sleep_val - 7)
    ax2.text(4.2, y_pred[-1], f'斜率={slope:.0f}', fontsize=8, color=color)

ax2.set_xlabel('咖啡（杯/天）', fontsize=11)
ax2.set_ylabel('工作效率指数', fontsize=11)
ax2.set_title('图1b：不同睡眠水平下，咖啡的边际效应不同', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.2)

plt.suptitle('图1：交互效应 — 咖啡的效果依赖于睡眠', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\009-interaction-coffee-sleep.png',
            dpi=200, bbox_inches='tight')
print("✅ 交互效应图已保存")

# ========== 5. 可视化二：边际效应图 ==========
fig, ax = plt.subplots(figsize=(9, 5))
sleep_range = np.linspace(4, 10, 100)
# 咖啡的边际效应 = β_coffee + β_interact * (sleep - 7)
marginal_effect = m_interact.params.iloc[1] + m_interact.params.iloc[3] * (sleep_range - 7)
# 标准误近似
se_marginal = np.sqrt(m_interact.bse.iloc[1]**2
                      + m_interact.bse.iloc[3]**2 * (sleep_range - 7)**2)

ax.plot(sleep_range, marginal_effect, color=C_MAIN, linewidth=2.5,
        label='咖啡的边际效应（每杯咖啡提升的效率）')
ax.fill_between(sleep_range,
                marginal_effect - 1.96 * se_marginal,
                marginal_effect + 1.96 * se_marginal,
                color=C_MAIN, alpha=0.12)

ax.axhline(y=0, color='gray', linewidth=1, linestyle='--')
ax.axvline(x=7, color=C_RED, linewidth=1.5, linestyle='--', alpha=0.6, label='睡眠 7h（临界点）')

ax.set_xlabel('睡眠时间（小时）', fontsize=12)
ax.set_ylabel('咖啡的边际效应', fontsize=12)
ax.set_title('图2：边际效应图 — 咖啡的效果如何随着睡眠变化', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\009-marginal-effect.png',
            dpi=200, bbox_inches='tight')
print("✅ 边际效应图已保存")

# ========== 6. 输出摘要 ==========
print(f"""
╔══════════════════════════════════════════════════════╗
║  交互效应模型结果                                     ║
╠══════════════════════════════════════════════════════╣
║  不带交互项: coffee β = {m_simple.params.iloc[1]:.1f} (sleep β = {m_simple.params.iloc[2]:.1f})   ║
║  带交互项:                                             ║
║    coffee 主效应     = {m_interact.params.iloc[1]:.1f}                     ║
║    sleep 主效应      = {m_interact.params.iloc[2]:.1f}                     ║
║    coffee×sleep 交互 = {m_interact.params.iloc[3]:.1f}                     ║
║  ─────────────────────────────────────────────────── ║
║  解释:                                                ║
║  当 sleep=7h 时, 每杯咖啡提升 {m_interact.params.iloc[1]:.0f} 单位效率        ║
║  睡眠每多 1h, 咖啡的效应额外增加 {m_interact.params.iloc[3]:.1f} 单位      ║
╚══════════════════════════════════════════════════════╝
""")
