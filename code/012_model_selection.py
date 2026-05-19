"""
知乎计量经济学系列 — 第 12 篇配套代码
主题：模型选择——少即是多（AIC/BIC）
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

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

# ========== 1. 模拟数据：房价预测的候选变量 ==========
np.random.seed(42)
n = 200

# 真实信号变量（4 个）
area = np.random.normal(120, 30, n)        # 面积（m²）
bedrooms = np.random.randint(1, 5, n)       # 卧室数
floor = np.random.randint(1, 30, n)         # 楼层
location_score = np.random.uniform(0, 10, n) # 地段评分（0-10）

# 噪声变量（8 个完全无关的随机变量）
noise_vars = {}
for i in range(8):
    noise_vars[f'noise_{i+1}'] = np.random.normal(0, 1, n)

# 真实房价生成（只有 4 个信号变量起作用）
price = (
    50 + 2.5 * area + 8 * bedrooms + 0.3 * floor + 15 * location_score
    + np.random.normal(0, 30, n)
)

# 组装 DataFrame
df = pd.DataFrame({
    'area': area,
    'bedrooms': bedrooms,
    'floor': floor,
    'location': location_score,
    'price': price,
    **noise_vars
})

print(f"✅ 模拟数据生成完成：{n} 条记录，4 个信号变量 + 8 个噪声变量")
print(f"   真实房价均值: {price.mean():.1f}, 标准差: {price.std():.1f}")

# ========== 2. 遍历所有可能的模型组合 ==========
all_vars = ['area', 'bedrooms', 'floor', 'location'] + [f'noise_{i+1}' for i in range(8)]
k_total = len(all_vars)

results = []
for n_vars in range(1, k_total + 1):
    for combo in combinations(range(k_total), n_vars):
        subset = [all_vars[i] for i in combo]
        X = sm.add_constant(df[subset])
        y = df['price']
        model = sm.OLS(y, X).fit()
        results.append({
            'n_vars': n_vars,
            'vars': subset,
            'r2': model.rsquared,
            'r2_adj': model.rsquared_adj,
            'aic': model.aic,
            'bic': model.bic,
            'model': model
        })

results_df = pd.DataFrame(results)
print(f"✅ 模型遍历完成：共评估 {len(results_df)} 个候选模型")

# ========== 图 1：AIC/BIC 对比（按模型复杂度分组） ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：AIC vs 模型数量（每个复杂度取最佳模型）
best_by_size = results_df.loc[results_df.groupby('n_vars')['aic'].idxmin()]

ax1 = axes[0]
ax1.plot(best_by_size['n_vars'], best_by_size['aic'],
         color=C_MAIN, marker='o', linewidth=2, markersize=8, label='AIC')
# 标记最小值
min_aic_idx = best_by_size['aic'].idxmin()
min_aic_row = best_by_size.loc[min_aic_idx]
ax1.scatter(min_aic_row['n_vars'], min_aic_row['aic'],
            color=C_RED, s=150, zorder=5, label=f'最佳 AIC ({int(min_aic_row["n_vars"])} 变量)')
ax1.annotate(f'AIC 最低\n{int(min_aic_row["n_vars"])} 个变量',
             xy=(min_aic_row['n_vars'], min_aic_row['aic']),
             xytext=(min_aic_row['n_vars'] + 1.5, min_aic_row['aic'] + 5),
             fontsize=10, color=C_RED, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.5))

ax1.set_xlabel('变量个数', fontsize=12)
ax1.set_ylabel('AIC', fontsize=12)
ax1.set_title('(a) AIC vs 模型复杂度', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.2)

# 右图：BIC vs 模型数量（每个复杂度取最佳模型）
best_by_size_bic = results_df.loc[results_df.groupby('n_vars')['bic'].idxmin()]

ax2 = axes[1]
ax2.plot(best_by_size_bic['n_vars'], best_by_size_bic['bic'],
         color=C_PURPLE, marker='s', linewidth=2, markersize=8, label='BIC')
# 标记最小值
min_bic_idx = best_by_size_bic['bic'].idxmin()
min_bic_row = best_by_size_bic.loc[min_bic_idx]
ax2.scatter(min_bic_row['n_vars'], min_bic_row['bic'],
            color=C_RED, s=150, zorder=5, label=f'最佳 BIC ({int(min_bic_row["n_vars"])} 变量)')
ax2.annotate(f'BIC 最低\n{int(min_bic_row["n_vars"])} 个变量',
             xy=(min_bic_row['n_vars'], min_bic_row['bic']),
             xytext=(min_bic_row['n_vars'] + 1.5, min_bic_row['bic'] + 10),
             fontsize=10, color=C_RED, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.5))

ax2.set_xlabel('变量个数', fontsize=12)
ax2.set_ylabel('BIC', fontsize=12)
ax2.set_title('(b) BIC vs 模型复杂度', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.2)

plt.suptitle('图 1：AIC / BIC 随模型复杂度的变化', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/012-model-comparison.png', dpi=200, bbox_inches='tight')
print("✅ 图 1 已保存：AIC/BIC 模型比较图")

# ========== 图 2：R² vs 复杂度的权衡曲线 ==========
fig2, ax = plt.subplots(figsize=(10, 6))

# 取每个复杂度下的最佳（最高 R²）模型
best_r2_by_size = results_df.loc[results_df.groupby('n_vars')['r2'].idxmax()]

ax.plot(best_r2_by_size['n_vars'], best_r2_by_size['r2'],
        color=C_MAIN, marker='o', linewidth=2.5, markersize=8, label='R²')
ax.plot(best_r2_by_size['n_vars'], best_r2_by_size['r2_adj'],
        color=C_ACCENT, marker='s', linewidth=2.5, markersize=8, label='调整 R²')

# 标记转折点——调整 R² 最大处
best_adj_idx = best_r2_by_size['r2_adj'].idxmax()
best_adj_row = best_r2_by_size.loc[best_adj_idx]
ax.axvline(x=best_adj_row['n_vars'], color=C_GRAY, linestyle='--', alpha=0.6)
ax.annotate(f'调整 R² 峰值\n{int(best_adj_row["n_vars"])} 个变量',
            xy=(best_adj_row['n_vars'], best_adj_row['r2_adj']),
            xytext=(best_adj_row['n_vars'] + 2, best_adj_row['r2_adj'] - 0.02),
            fontsize=10, color=C_ACCENT, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_ACCENT, lw=1.5))

# 标记过拟合区域
ax.axvspan(8, 12, alpha=0.08, color=C_RED, label='过拟合风险区域')

ax.set_xlabel('变量个数', fontsize=12)
ax.set_ylabel('R² / 调整 R²', fontsize=12)
ax.set_title('图 2：R² 与模型复杂度的权衡——少即是多', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(alpha=0.2)
ax.set_ylim(0.5, 1.0)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/012-tradeoff.png', dpi=200, bbox_inches='tight')
print("✅ 图 2 已保存：R² vs 复杂度权衡曲线")

# ========== 3. 前向选择演示 ==========
print("\n" + "=" * 60)
print("前向选择过程：")
print("=" * 60)

remaining = all_vars.copy()
selected = []
forward_steps = []

for step in range(6):  # 演示前 6 步
    best_aic = np.inf
    best_var = None
    best_model = None

    for var in remaining:
        candidate = selected + [var]
        X = sm.add_constant(df[candidate])
        model = sm.OLS(df['price'], X).fit()
        if model.aic < best_aic:
            best_aic = model.aic
            best_var = var
            best_model = model

    selected.append(best_var)
    remaining.remove(best_var)
    forward_steps.append({
        'step': step + 1,
        'var_added': best_var,
        'aic': best_model.aic,
        'bic': best_model.bic,
        'r2_adj': best_model.rsquared_adj,
        'model': best_model
    })
    sig = "✅ 信号" if 'noise' not in best_var else "⚠️ 噪声"
    print(f"  步骤 {step+1}: 添加 {best_var:12s} | AIC={best_model.aic:.1f} | "
          f"BIC={best_model.bic:.1f} | 调整R²={best_model.rsquared_adj:.4f} {sig}")

forward_df = pd.DataFrame(forward_steps)

# ========== 图 3：前向选择步骤可视化 ==========
fig3, ax3 = plt.subplots(figsize=(11, 6))

steps = forward_df['step']
aic_vals = forward_df['aic']
bic_vals = forward_df['bic']
r2adj_vals = forward_df['r2_adj']

# 双 Y 轴
color_aic = C_MAIN
color_bic = C_PURPLE

ax3.plot(steps, aic_vals, color=color_aic, marker='o', linewidth=2.5,
         markersize=10, label='AIC')
ax3.plot(steps, bic_vals, color=color_bic, marker='s', linewidth=2.5,
         markersize=10, label='BIC')

# 标注每一步添加的变量名
for i, row in forward_df.iterrows():
    label = row['var_added'].replace('_', ' ')
    offset_y = -8 if i % 2 == 0 else 8
    ax3.annotate(label, xy=(row['step'], row['aic']),
                 xytext=(row['step'], row['aic'] + offset_y),
                 fontsize=8, ha='center', fontstyle='italic',
                 color=C_GRAY, rotation=0)

ax3.set_xlabel('前向选择步骤', fontsize=12)
ax3.set_ylabel('AIC / BIC', fontsize=12)
ax3.set_title('图 3：前向选择——每一步添加使 AIC/BIC 下降最多的变量', fontsize=13, fontweight='bold')
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(alpha=0.2)
ax3.set_xticks(steps)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/012-forward-selection.png', dpi=200, bbox_inches='tight')
print("✅ 图 3 已保存：前向选择步骤图")

# ========== 4. 最佳模型结果 ==========
print("\n" + "=" * 60)
print("最佳模型（AIC 最小）结果：")
print("=" * 60)
best_model_row = results_df.loc[results_df['aic'].idxmin()]
print(f"  变量个数: {best_model_row['n_vars']}")
print(f"  选择变量: {best_model_row['vars']}")
print(f"  R²:       {best_model_row['r2']:.4f}")
print(f"  调整 R²:  {best_model_row['r2_adj']:.4f}")
print(f"  AIC:      {best_model_row['aic']:.2f}")
print(f"  BIC:      {best_model_row['bic']:.2f}")
print()

# 输出前 5 个最佳 AIC 模型
top5_aic = results_df.sort_values('aic').head(5)
print("AIC 前 5 模型：")
for i, row in top5_aic.iterrows():
    print(f"  {row['n_vars']} 变量 | AIC={row['aic']:.1f} | BIC={row['bic']:.1f} | "
          f"R²={row['r2']:.3f} | 变量: {', '.join(row['vars'][:5])}")

print("\n🎉 第 12 篇所有图表已生成完毕！")
