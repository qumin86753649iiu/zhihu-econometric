"""
知乎计量经济学系列 — 第 2 篇配套代码
主题：最小二乘法的几何直觉 — 用外卖配送费为例
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ========== 统一配置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
C_MAIN = '#2E86AB'
C_ACCENT = '#F18F01'
C_RED = '#C73E1D'
C_GREEN = '#4C9F70'

# ========== 1. 生成模拟数据 ==========
np.random.seed(2024)
n = 20
distance = np.random.uniform(0.5, 10, n)  # 公里
# 真实关系：配送费 = 3元底价 + 2元/公里 + 随机波动
fee = 3.0 + 2.0 * distance + np.random.normal(0, 2.5, n)
fee = np.maximum(fee, 4.0)  # 配送费不低于4元

df = pd.DataFrame({'distance_km': distance, 'fee_yuan': fee})

# ========== 2. 三个候选拟合线的对比图 ==========
candidates = [
    (0.5, 3.5, '斜线 A：太平缓'),
    (5.0, 1.5, '斜线 B：太陡峭'),
    (3.7, 2.0, '斜线 C：刚合适'),
]

x_line = np.linspace(0, 10.5, 100)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
colors = ['#A23B72', '#F18F01', '#4C9F70']

for ax, (b0, b1, label) in zip(axes, candidates):
    ax.scatter(df['distance_km'], df['fee_yuan'], alpha=0.7,
               color=C_MAIN, edgecolors='white', linewidth=0.5, s=55)
    y_line = b0 + b1 * x_line
    ax.plot(x_line, y_line, color=colors[0], linewidth=2, label=label)
    # 画残差竖线
    y_pred = b0 + b1 * df['distance_km']
    for i in range(len(df)):
        ax.plot([df['distance_km'].iloc[i], df['distance_km'].iloc[i]],
                [df['fee_yuan'].iloc[i], y_pred.iloc[i]],
                color='gray', linewidth=0.8, alpha=0.4)
    ax.set_xlabel('配送距离（公里）', fontsize=10)
    if ax == axes[0]:
        ax.set_ylabel('配送费（元）', fontsize=10)
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 28)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

plt.suptitle('图1：不同斜率的拟合线 — 哪条更合适？', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\002-candidate-lines.png',
            dpi=200, bbox_inches='tight')
print("✅ 候选线对比图已保存")

# ========== 3. 展示"误差平方"的可视化 ==========
# 选"太缓"那条线，画出每个点的误差正方形区域
fig, ax = plt.subplots(figsize=(9, 5.5))

b0_bad, b1_bad = 0.5, 3.5
y_pred_bad = b0_bad + b1_bad * df['distance_km']
residuals = df['fee_yuan'] - y_pred_bad

ax.scatter(df['distance_km'], df['fee_yuan'], alpha=0.65,
           color=C_MAIN, edgecolors='white', linewidth=0.5, s=60,
           label='真实数据点')
x_line = np.linspace(0, 10.5, 100)
ax.plot(x_line, b0_bad + b1_bad * x_line,
        color='#A23B72', linewidth=2, linestyle='--', label='拟合线 (太平缓)')

# 画每个点的误差（纵线）和"面积"
for i in range(len(df)):
    xi = df['distance_km'].iloc[i]
    yi = df['fee_yuan'].iloc[i]
    y_hat = y_pred_bad.iloc[i]
    err = residuals.iloc[i]
    
    # 纵线
    ax.plot([xi, xi], [yi, y_hat], color='#C73E1D', linewidth=1.2, alpha=0.5)
    
    # 标误差值
    ax.annotate(f'{err:.1f}', (xi + 0.15, (yi + y_hat)/2),
                fontsize=7, color='#C73E1D', alpha=0.8)

ax.set_xlabel('配送距离（公里）', fontsize=12)
ax.set_ylabel('配送费（元）', fontsize=12)
ax.set_title('图2：误差 = 真实值 - 预测值（垂直距离）', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.25)

# 左下角加一个注解框
rss_bad = np.sum(residuals**2)
ax.text(0.5, 26, f'误差平方和 (SSE) = {rss_bad:.0f}',
        fontsize=11, bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEDE1', alpha=0.8))

plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\002-residuals.png',
            dpi=200, bbox_inches='tight')
print("✅ 误差可视化图已保存")

# ========== 4. 最佳拟合线 + 误差平方展示 ==========
fig, ax = plt.subplots(figsize=(9, 5.5))

# 用 OLS 找最佳线
X = sm.add_constant(df['distance_km'])
model = sm.OLS(df['fee_yuan'], X).fit()
b0_best, b1_best = model.params['const'], model.params['distance_km']
y_pred_best = b0_best + b1_best * df['distance_km']
residuals_best = df['fee_yuan'] - y_pred_best

ax.scatter(df['distance_km'], df['fee_yuan'], alpha=0.65,
           color=C_MAIN, edgecolors='white', linewidth=0.5, s=60,
           label='真实数据点')
ax.plot(x_line, b0_best + b1_best * x_line,
        color='#C73E1D', linewidth=2.5, label=f'最佳拟合线 (y = {b0_best:.1f} + {b1_best:.1f}x)')

# 画误差平方的示意（用小方块）
for i in range(0, len(df)):
    xi = df['distance_km'].iloc[i]
    yi = df['fee_yuan'].iloc[i]
    y_hat = y_pred_best.iloc[i]
    err = residuals_best.iloc[i]
    side = abs(err)
    # 用半透明矩形表示"平方"面积
    rect = plt.Rectangle((xi, min(yi, y_hat)), side, side,
                          facecolor='#F18F01', alpha=0.08, linewidth=0)
    ax.add_patch(rect)

# 标注
rss_best = np.sum(residuals_best**2)
ax.text(0.5, 26, f'误差平方和 (SSE) = {rss_best:.0f}  ← 最小！',
        fontsize=11, color='#C73E1D',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEDE1', alpha=0.8))

ax.set_xlabel('配送距离（公里）', fontsize=12)
ax.set_ylabel('配送费（元）', fontsize=12)
ax.set_title('图3：OLS 找到的线 — 误差平方和最小', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\002-ols-best.png',
            dpi=200, bbox_inches='tight')
print("✅ OLS 最佳拟合线图已保存")

# ========== 5. 对比：坏线 vs 好线 ==========
print(f"""
╔═══════════════════════════════════════════╗
║  模型对比                                 ║
╠═══════════════════════════════════════════╣
║  真实关系:   fee = 3.0 + 2.0 x km        ║
╠═══════════════════════════════════════════╣
║  坏线 A:    SSE = {rss_bad:.0f}               ║
║  最佳 OLS:  SSE = {rss_best:.0f}               ║
║  OLS 系数:  β₀ = {b0_best:.1f}, β₁ = {b1_best:.2f}    ║
╚═══════════════════════════════════════════╝
""")

print("\n=== 完整回归结果 ===")
print(model.summary())
