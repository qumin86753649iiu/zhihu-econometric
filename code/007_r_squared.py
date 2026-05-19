"""
知乎计量经济学系列 — 第 7 篇配套代码
主题：R² 和调整 R² — 模型评价与过度拟合
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

# ========== 1. 模拟房价数据 ==========
np.random.seed(42)
n = 100
area = np.random.uniform(40, 180, n)
location = np.random.uniform(1, 10, n)
age = np.random.uniform(0, 30, n)
bedrooms = np.random.randint(1, 6, n)
price = (30 + 2.8 * area + 5 * location
         - 1.5 * age + 8 * bedrooms
         + np.random.normal(0, 40, n))
price = np.maximum(price, 30)

df = pd.DataFrame({
    'price': price, 'area': area, 'location': location,
    'age': age, 'bedrooms': bedrooms
})

# ========== 2. 逐步添加变量，跟踪 R² 和调整 R² ==========
variables = ['area', 'location', 'age', 'bedrooms']
results = []

for k in range(1, len(variables) + 1):
    X = sm.add_constant(df[variables[:k]])
    m = sm.OLS(df['price'], X).fit()
    results.append({
        'vars': '+'.join(variables[:k]),
        'k': k + 1,  # +1 截距
        'r2': m.rsquared,
        'adj_r2': m.rsquared_adj,
        'aic': m.aic,
        'bic': m.bic
    })

df_results = pd.DataFrame(results)

# ========== 3. 加入随机噪声变量，展示 R² 的膨胀 ==========
np.random.seed(123)
noise_results = []
var_list = ['area', 'location', 'age', 'bedrooms']

# 从真实变量开始，逐步添加无意义的噪声列
for i in range(2, 22):
    X_cols = ['area', 'location', 'age', 'bedrooms']
    for j in range(i - 4):  # 添加噪声变量
        df[f'noise_{j}'] = np.random.normal(0, 1, n)
        X_cols.append(f'noise_{j}')
    X = sm.add_constant(df[X_cols])
    m = sm.OLS(df['price'], X).fit()
    noise_results.append({
        'n_vars': i,
        'n_noise': i - 4,
        'r2': m.rsquared,
        'adj_r2': m.rsquared_adj
    })

df_noise = pd.DataFrame(noise_results)

# ========== 4. 画图一：R² vs 调整 R² 随变量增加的变化 ==========
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左图：真实变量
ax1 = axes[0]
k_vals = df_results['k']
ax1.plot(k_vals, df_results['r2'], 'o-', color=C_MAIN, linewidth=2.5, markersize=8, label='R²')
ax1.plot(k_vals, df_results['adj_r2'], 's-', color=C_ACCENT, linewidth=2.5, markersize=8, label='调整 R²')
for i, row in df_results.iterrows():
    ax1.annotate(row['vars'], (row['k'], row['r2']), fontsize=8,
                 textcoords="offset points", xytext=(0, 12), ha='center')
ax1.set_xticks(k_vals)
ax1.set_xlabel('模型中变量数量（含截距）', fontsize=11)
ax1.set_ylabel('R² / 调整 R²', fontsize=11)
ax1.set_title('图1a：添加真实变量 — 两者同步上升', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.2)
ax1.set_ylim(0.75, 0.95)

# 右图：加入噪声变量
ax2 = axes[1]
ax2.plot(df_noise['n_vars'], df_noise['r2'], 'o-', color=C_MAIN, linewidth=2, markersize=6, label='R²（不断上升）')
ax2.plot(df_noise['n_vars'], df_noise['adj_r2'], 's-', color=C_RED, linewidth=2, markersize=6, label='调整 R²（停滞甚至下降）')
ax2.axvline(x=4, color='gray', linewidth=1.5, linestyle='--', alpha=0.6, label='真实变量边界')
ax2.set_xlabel('模型中变量总数', fontsize=11)
ax2.set_ylabel('R² / 调整 R²', fontsize=11)
ax2.set_title('图1b：加入无用变量 — R² 虚胖，调整 R² 诚实', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.2)

plt.suptitle('图1：R² vs 调整 R² — 为什么不能只看 R²', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\007-r2-comparison.png',
            dpi=200, bbox_inches='tight')
print("✅ R² 对比图已保存")

# ========== 5. 两种模型的直观对比 ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 模型 1：只用面积（简单）
m1 = sm.OLS(df['price'], sm.add_constant(df['area'])).fit()
# 模型 2：面积+位置+房龄+卧室（合理）
m2 = sm.OLS(df['price'], sm.add_constant(df[['area', 'location', 'age', 'bedrooms']])).fit()
# 模型 3：全部+20个噪声变量（过拟合）
X3 = df[['area', 'location', 'age', 'bedrooms']].copy()
for j in range(20):
    X3[f'noise_{j}'] = np.random.normal(0, 1, n)
m3 = sm.OLS(df['price'], sm.add_constant(X3)).fit()

# 拟合值 vs 真实值
for ax, model, title, color in [
    (axes[0], m1, '简单模型 (仅面积)', C_MAIN),
    (axes[1], m3, '过拟合模型 (面积+20个噪声)', C_RED)]:
    
    ax.scatter(model.fittedvalues, df['price'], alpha=0.5, color=color,
               edgecolors='white', linewidth=0.5, s=40)
    ax.plot([50, 650], [50, 650], '--', color='gray', linewidth=1.5)
    ax.set_xlabel('预测值（万元）', fontsize=10)
    ax.set_ylabel('真实值（万元）', fontsize=10)
    ax.set_title(f'{title}\nR²={model.rsquared:.3f}, 调整R²={model.rsquared_adj:.3f}',
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.2)
    ax.set_xlim(50, 650)
    ax.set_ylim(50, 650)

plt.suptitle('图2：简单 vs 过拟合 — 预测值和真实值的差异', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\007-overfit-vs-simple.png',
            dpi=200, bbox_inches='tight')
print("✅ 过拟合对比图已保存")

# ========== 6. 输出摘要 ==========
print(f"""
╔══════════════════════════════════════════════════════╗
║  模型对比                                             ║
╠══════════════════════════════════════════════════════╣
║  模型                    变量数    R²      调整 R²    ║
║  ─────────────────────────────────────────────────── ║
║  简单 (仅面积)              2    {m1.rsquared:.3f}    {m1.rsquared_adj:.3f}      ║
║  合理 (4个真实变量)          5    {m2.rsquared:.3f}    {m2.rsquared_adj:.3f}      ║
║  过拟合 (4真实+20噪声)     25    {m3.rsquared:.3f}    {m3.rsquared_adj:.3f}      ║
╚══════════════════════════════════════════════════════╝
""")
