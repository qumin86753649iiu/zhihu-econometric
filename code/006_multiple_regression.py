"""
知乎计量经济学系列 — 第 6 篇配套代码
主题：多元回归 — 控制其他变量，看"净效应"
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
C_MAIN = '#2E86AB'
C_ACCENT = '#F18F01'
C_RED = '#C73E1D'
C_GREEN = '#4C9F70'

# ========== 1. 生成和上一篇文章一样的房价数据 ==========
np.random.seed(42)
n = 200
area = np.random.uniform(40, 180, n)
bedrooms = np.random.randint(1, 6, n)
floor = np.random.randint(1, 35, n)
location = np.random.uniform(1, 10, n)
age = np.random.uniform(0, 30, n)
price = (30 + 2.8 * area + 5 * location
         - 1.5 * age + 8 * bedrooms
         + np.random.normal(0, 40, n))
price = np.maximum(price, 30)

df = pd.DataFrame({
    'price': price, 'area': area, 'bedrooms': bedrooms,
    'floor': floor, 'location': location, 'age': age
})

# ========== 2. 3D 散点图 + 回归平面 ==========
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 选两个变量画 3D：面积和位置
ax.scatter(df['area'], df['location'], df['price'],
           alpha=0.5, color=C_MAIN, s=30, edgecolors='white', linewidth=0.5)

# 回归平面
X_3d = sm.add_constant(df[['area', 'location']])
m_3d = sm.OLS(df['price'], X_3d).fit()
area_range = np.linspace(40, 180, 20)
loc_range = np.linspace(1, 10, 20)
A, L = np.meshgrid(area_range, loc_range)
P = m_3d.params.iloc[0] + m_3d.params.iloc[1] * A + m_3d.params.iloc[2] * L
ax.plot_surface(A, L, P, alpha=0.25, color=C_ACCENT, edgecolor='none')

ax.set_xlabel('面积 (m²)', fontsize=10)
ax.set_ylabel('位置评分', fontsize=10)
ax.set_zlabel('房价 (万元)', fontsize=10)
ax.set_title('图1：多元回归的"平面"——面积和位置共同解释房价', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\006-3d-plane.png',
            dpi=200, bbox_inches='tight')
print("✅ 3D 回归平面图已保存")

# ========== 3. 简单 vs 多元回归系数对比 ==========
# 简单回归（只有面积）
m_simple_area = sm.OLS(df['price'], sm.add_constant(df['area'])).fit()
# 简单回归（只有位置）
m_simple_loc = sm.OLS(df['price'], sm.add_constant(df['location'])).fit()
# 多元回归
X_multi = sm.add_constant(df[['area', 'location', 'age', 'bedrooms', 'floor']])
m_multi = sm.OLS(df['price'], X_multi).fit()

print(f"\n=== 简单 vs 多元系数对比 ===")
print(f"面积 — 简单回归: {m_simple_area.params.iloc[1]:.2f}")
print(f"面积 — 多元回归: {m_multi.params.iloc[1]:.2f}")
print(f"位置 — 简单回归: {m_simple_loc.params.iloc[1]:.2f}")
print(f"位置 — 多元回归: {m_multi.params.iloc[2]:.2f}")

# 可视化对比
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, var_name, simple_model, multi_beta, multi_se in [
    (axes[0], '面积', m_simple_area, m_multi.params.iloc[1], m_multi.bse.iloc[1]),
    (axes[1], '位置', m_simple_loc, m_multi.params.iloc[2], m_multi.bse.iloc[2])]:
    
    simple_beta = simple_model.params.iloc[1]
    simple_se = simple_model.bse.iloc[1]
    
    labels = ['简单回归\n(遗漏变量)', '多元回归\n(控制其他)']
    betas = [simple_beta, multi_beta]
    errs = [1.96 * simple_se, 1.96 * multi_se]
    colors_bar = ['#A23B72', C_GREEN]
    
    bars = ax.bar(labels, betas, yerr=errs, capsize=5, color=colors_bar,
                  alpha=0.75, edgecolor='white', width=0.5)
    ax.axhline(y=0, color='gray', linewidth=0.8)
    ax.set_title(f'{var_name}的效应：简单 vs 多元', fontsize=12, fontweight='bold')
    ax.set_ylabel('系数（万元/单位）', fontsize=10)
    ax.grid(axis='y', alpha=0.2)
    
    # 标注数值
    for bar, beta in zip(bars, betas):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 2 if beta > 0 else bar.get_height() - 8,
                f'{beta:.1f}', ha='center', fontsize=10, fontweight='bold')

plt.suptitle('图2：控制 vs 不控制其他变量——遗漏变量偏误的可视化', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\006-simple-vs-multi.png',
            dpi=200, bbox_inches='tight')
print("✅ 简单 vs 多元对比图已保存")

# ========== 4. 偏回归图——控制后的"净关系" ==========
# 对"面积"做偏回归：先去除其他变量的影响
X_others = sm.add_constant(df[['location', 'age', 'bedrooms', 'floor']])
resid_price = sm.OLS(df['price'], X_others).fit().resid  # 价格中不能被其他变量解释的部分
resid_area = sm.OLS(df['area'], X_others).fit().resid    # 面积中不能被其他变量解释的部分

m_partial = sm.OLS(resid_price, sm.add_constant(resid_area)).fit()

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.scatter(resid_area, resid_price, alpha=0.5, color=C_MAIN, edgecolors='white', linewidth=0.5, s=40)
xp = np.linspace(resid_area.min(), resid_area.max(), 50)
yp = m_partial.params.iloc[0] + m_partial.params.iloc[1] * xp
ax.plot(xp, yp, color=C_RED, linewidth=2.5,
        label=f'偏回归斜率 = {m_partial.params.iloc[1]:.2f}')

ax.axhline(y=0, color='gray', linewidth=0.8, linestyle=':')
ax.axvline(x=0, color='gray', linewidth=0.8, linestyle=':')
ax.set_xlabel('面积的"净变异"（控制了位置、房龄、卧室、楼层后）', fontsize=11)
ax.set_ylabel('房价的"净变异"（控制了位置、房龄、卧室、楼层后）', fontsize=11)
ax.set_title('图3：偏回归图 — 控制其他变量后，面积对房价的"净效应"', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\006-partial-regression.png',
            dpi=200, bbox_inches='tight')
print("✅ 偏回归图已保存")

# ========== 5. 完整结果摘要 ==========
print(f"""
╔══════════════════════════════════════════════════════╗
║  多元回归 vs 简单回归的差异                            ║
╠══════════════════════════════════════════════════════╣
║                    简单回归    多元回归    变化         ║
║  ──────────────────────────────────────────────── ║
║  面积 (万/m²)     {m_simple_area.params.iloc[1]:>8.2f}  {m_multi.params.iloc[1]:>8.2f}  {m_multi.params.iloc[1]-m_simple_area.params.iloc[1]:>+7.2f}  ║
║  位置 (万/分)     {m_simple_loc.params.iloc[1]:>8.2f}  {m_multi.params.iloc[2]:>8.2f}  {m_multi.params.iloc[2]-m_simple_loc.params.iloc[1]:>+7.2f}  ║
║  R²              {m_simple_area.rsquared:>8.3f}  {m_multi.rsquared:>8.3f}              ║
╚══════════════════════════════════════════════════════╝
""")
