"""
知乎计量经济学系列 — 第 8 篇配套代码
主题：虚拟变量 — 把分类变量放进回归
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

# ========== 1. 生成数据：房子的朝向模拟 ==========
np.random.seed(42)
n = 200

area = np.random.uniform(50, 150, n)
location = np.random.uniform(1, 10, n)
age = np.random.uniform(0, 25, n)

# 朝向（分类变量）：南、北、东、西
orientations = ['南', '北', '东', '西']
orientation = np.random.choice(orientations, n, p=[0.35, 0.25, 0.25, 0.15])

# 真实效应：朝南+10万，朝北-5万，朝东+0，朝西-5万
orient_effects = {'南': 10, '北': -5, '东': 0, '西': -5}
orient_numeric = [orient_effects[o] for o in orientation]

price = (30 + 2.5 * area + 5 * location
         - 1.2 * age + np.array(orient_numeric)
         + np.random.normal(0, 25, n))
price = np.maximum(price, 30)

df = pd.DataFrame({
    'price': price, 'area': area, 'location': location,
    'age': age, 'orientation': orientation
})

# ========== 2. 创建虚拟变量 ==========
# pandas 一行搞定：drop_first=True 自动规避虚拟变量陷阱
df_dummies = pd.get_dummies(df['orientation'], prefix='朝向', drop_first=True).astype(int)
df_reg = pd.concat([df[['price', 'area', 'location', 'age']], df_dummies], axis=1)

print("=== 前5行数据 ===")
print(df.head())
print("\n=== 虚拟变量（前5行）===")
print(df_dummies.head())

# ========== 3. 跑回归 ==========
X = sm.add_constant(df_reg.drop('price', axis=1))
m = sm.OLS(df['price'], X).fit()

print("\n=== 回归结果 ===")
print(m.summary())

# ========== 4. 可视化：朝南 vs 其他 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：不同朝向的平均房价（原始数据）
ax1 = axes[0]
means = df.groupby('orientation')['price'].mean().reindex(['南', '北', '东', '西'])
colors_orient = [C_MAIN, C_ACCENT, C_RED, C_GREEN]
bars = ax1.bar(means.index, means.values, color=colors_orient, alpha=0.75, edgecolor='white', width=0.5)
for bar, val in zip(bars, means.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             f'{val:.0f}万', ha='center', fontsize=10, fontweight='bold')
ax1.set_ylabel('平均房价（万元）', fontsize=11)
ax1.set_title('图1a：不同朝向的平均房价（原始数据）', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.2)

# 右图：控制面积、位置、房龄后，各朝向的净效应
ax2 = axes[1]
# 系数 = 相对于基准组（朝向"南"，因为drop_first=True会去掉第一个）
# 需要手动找出基准组
coefs = {'南': 0}  # 基准组效应=0
for i, orient_name in enumerate(['北', '东', '西']):
    col_name = f'朝向_{orient_name}'
    if col_name in m.params:
        coefs[orient_name] = m.params[col_name]

labels = list(coefs.keys())
values = list(coefs.values())
colors_bar = [C_MAIN, C_ACCENT, C_RED, C_GREEN]

bars2 = ax2.bar(labels, values, color=colors_bar, alpha=0.75, edgecolor='white', width=0.5)
for bar, val in zip(bars2, values):
    y_pos = bar.get_height() + 1.5 if val >= 0 else bar.get_height() - 4
    ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
             f'{val:+.0f}万', ha='center', fontsize=10, fontweight='bold')

ax2.axhline(y=0, color='gray', linewidth=0.8)
ax2.set_ylabel('相对于朝南的房价差异（万元）', fontsize=11)
ax2.set_title('图1b：控制其他变量后，各朝向的"净效应"', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.2)
# 标注基准组
ax2.text(0, 0.5, '基准组', ha='center', fontsize=9, color='gray', fontstyle='italic')

plt.suptitle('图1：不同朝向对房价的影响 — 原始数据 vs 控制后', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\008-dummy-orientation.png',
            dpi=200, bbox_inches='tight')
print("✅ 虚拟变量对比图已保存")

# ========== 5. 虚拟变量陷阱演示 ==========
# 如果全部放进去会发生什么？
try:
    X_trap = sm.add_constant(pd.get_dummies(df['orientation'], prefix='朝向', drop_first=False))
    X_trap = pd.concat([X_trap, df[['area']]], axis=1)
    m_trap = sm.OLS(df['price'], X_trap).fit()
    print("\n=== 虚拟变量陷阱（未drop_first）===")
    print("注意看：标准误会很大，结果不可靠")
    print(m_trap.summary())
except Exception as e:
    print(f"\n虚拟变量陷阱导致完全多重共线性: {e}")

# ========== 6. 结果摘要 ==========
print(f"""
╔═══════════════════════════════════════════════╗
║  虚拟变量回归结果                               ║
╠═══════════════════════════════════════════════╣
║  基准组: 朝南 (效应设为0)                     ║
╠═══════════════════════════════════════════════╣
║  变量        系数      t值     p值   解释        ║
║  ──────────────────────────────────────────── ║
║  面积        {m.params['area']:>6.2f}   {m.tvalues['area']:>5.1f}  0.000  每+1m² → 价格+{m.params['area']:.1f}万   ║
║  位置        {m.params['location']:>6.2f}   {m.tvalues['location']:>5.1f}  0.000  每+1分 → 价格+{m.params['location']:.1f}万   ║
║  房龄        {m.params['age']:>6.2f}   {m.tvalues['age']:>5.1f}  0.000  每+1年 → 价格{m.params['age']:.1f}万   ║
""")

# 打印虚拟变量系数
for orient_name in ['北', '东', '西']:
    col = f'朝向_{orient_name}'
    if col in m.params:
        print(f"  朝向-{orient_name}    {m.params[col]:>6.2f}   {m.tvalues[col]:>5.1f}  {m.pvalues[col]:.3f}  相对于朝南{m.params[col]:+.1f}万")

print("╚═══════════════════════════════════════════════╝")
