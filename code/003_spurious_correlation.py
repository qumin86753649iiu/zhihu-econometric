"""
知乎计量经济学系列 — 第 3 篇配套代码
主题：相关不等于因果 — 模拟"假相关"和因果图
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
C_MAIN = '#2E86AB'
C_ACCENT = '#F18F01'
C_RED = '#C73E1D'
C_GREEN = '#4C9F70'

# ========== 1. 模拟"假相关"：两个完全无关的时间序列 ==========
np.random.seed(2023)
t = np.arange(1, 101)

# 完全独立的两个随机游走
series_a = np.cumsum(np.random.normal(0, 1, 100)) + 100  # 假装是"某国人均巧克力消费"
series_b = np.cumsum(np.random.normal(0, 1, 100)) + 50   # 假装是"某诺贝尔奖得主数量"

df_fake = pd.DataFrame({
    'year': t + 2000,
    'chocolate': series_a,
    'nobel': series_b
})

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 左上一：时间序列对比
ax1 = axes[0, 0]
ax1.plot(df_fake['year'], df_fake['chocolate'], color=C_MAIN, linewidth=2, label='巧克力消费指数')
ax1.plot(df_fake['year'], df_fake['nobel'], color=C_ACCENT, linewidth=2, label='诺贝尔奖指数')
ax1.set_xlabel('年份')
ax1.set_ylabel('指数')
ax1.set_title('两个随机序列 — 走势惊人相似？', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.2)

# 右上二：散点图 + 回归线
ax2 = axes[0, 1]
X = sm.add_constant(df_fake['chocolate'])
model = sm.OLS(df_fake['nobel'], X).fit()
ax2.scatter(df_fake['chocolate'], df_fake['nobel'], alpha=0.5, color=C_MAIN, s=30)
x_line = np.linspace(df_fake['chocolate'].min(), df_fake['chocolate'].max(), 100)
X_line = sm.add_constant(x_line)
y_pred = model.predict(X_line)
ax2.plot(x_line, y_pred, color=C_RED, linewidth=2, label=f'R² = {model.rsquared:.2f}')
ax2.set_xlabel('巧克力消费指数')
ax2.set_ylabel('诺贝尔奖指数')
ax2.set_title(f'看起来有相关？R² = {model.rsquared:.2f}', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.2)

# 左下三：残差图
ax3 = axes[1, 0]
residuals = model.resid
ax3.scatter(model.fittedvalues, residuals, alpha=0.5, color=C_ACCENT, s=30)
ax3.axhline(y=0, color='gray', linewidth=1, linestyle='--')
ax3.set_xlabel('拟合值')
ax3.set_ylabel('残差')
ax3.set_title('残差图：没有明显模式', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.2)

# 右下四：说明文字
ax4 = axes[1, 1]
ax4.axis('off')
text = (
    '⚠️ 这是两个完全独立的随机游走序列！\n\n'
    '它们之间没有任何因果关系。\n\n'
    '只是因为都有「趋势」，\n'
    '回归就显示出了「显著」的相关性。\n\n'
    f'R² = {model.rsquared:.2f}\n'
    f'P值 = {model.f_pvalue:.6f}\n\n'
    '如果只看结果报表，\n'
    '你会以为「巧克力消费」\n'
    '能预测诺贝尔奖——\n\n'
    '这就是伪回归 (Spurious Regression)。'
)
ax4.text(0.05, 0.95, text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFEDE1', alpha=0.9))

plt.suptitle('图1：伪回归 — 两个完全无关的序列也能跑出"显著"结果', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\003-spurious-regression.png',
            dpi=200, bbox_inches='tight')
print(f"✅ 伪回归图已保存 (R²={model.rsquared:.2f}, p={model.f_pvalue:.6f})")

# ========== 2. 因果图 (DAG) ==========
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# DAG 1: 反向因果
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# 节点
ax.annotate('刷手机', xy=(4, 5), fontsize=14, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2E86AB', edgecolor='none', alpha=0.8),
            color='white', fontweight='bold')
ax.annotate('感觉累', xy=(8, 5), fontsize=14, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#A23B72', edgecolor='none', alpha=0.8),
            color='white', fontweight='bold')

# 箭头
ax.annotate('', xy=(7.5, 5), xytext=(4.8, 5),
            arrowprops=dict(arrowstyle='->', color=C_RED, linewidth=2))
ax.text(6.2, 5.6, '还是', fontsize=11, color=C_RED, ha='center')
ax.annotate('', xy=(4.8, 4), xytext=(7.5, 4),
            arrowprops=dict(arrowstyle='->', color=C_ACCENT, linewidth=2, linestyle='dashed'))
ax.text(6.2, 3.4, '或者？', fontsize=11, color=C_ACCENT, ha='center')
ax.set_title('① 反向因果\n累才刷手机 vs 刷手机才累', fontsize=12, fontweight='bold')

# DAG 2: 遗漏变量
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.annotate('刷手机', xy=(2, 5), fontsize=14, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2E86AB', edgecolor='none', alpha=0.8),
            color='white', fontweight='bold')
ax.annotate('感觉累', xy=(8, 5), fontsize=14, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#A23B72', edgecolor='none', alpha=0.8),
            color='white', fontweight='bold')
ax.annotate('压力事件', xy=(5, 8.5), fontsize=14, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#F18F01', edgecolor='none', alpha=0.8),
            color='white', fontweight='bold')

ax.annotate('', xy=(5.5, 7.8), xytext=(3, 6),
            arrowprops=dict(arrowstyle='->', color='#4C9F70', linewidth=2))
ax.annotate('', xy=(7.5, 6), xytext=(6, 7.8),
            arrowprops=dict(arrowstyle='->', color='#4C9F70', linewidth=2))
ax.set_title('② 遗漏变量\n压力让人刷手机也让人累', fontsize=12, fontweight='bold')

# DAG 3: 纯噪声
ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.annotate('奶茶销量\n(今天)', xy=(5, 6), fontsize=14, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2E86AB', edgecolor='none', alpha=0.8),
            color='white', fontweight='bold')
ax.annotate('降雨概率\n(明天)', xy=(5, 2), fontsize=14, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#A23B72', edgecolor='none', alpha=0.8),
            color='white', fontweight='bold')
ax.annotate('┌─────────────┐\n│ 没有任何关系  │\n│ 碰巧相关而已  │\n└─────────────┘', xy=(8, 4), fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8E8E8', edgecolor='gray', alpha=0.8))
ax.set_title('③ 纯噪声\nN 次尝试中总会有一次巧合', fontsize=12, fontweight='bold')

plt.suptitle('图2：三种"假相关"的因果路径图', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\003-dag-causality.png',
            dpi=200, bbox_inches='tight')
print("✅ DAG 因果图已保存")

# ========== 3. 模拟"遗漏变量"效应 ==========
np.random.seed(42)
n = 100

# 真实的因果结构：压力 (Z) → 刷手机 (X) 和 睡眠质量 (Y)
stress = np.random.normal(0, 1, n)  # 不可观测的压力水平
phone_hours = 3 + 1.5 * stress + np.random.normal(0, 0.5, n)  # 刷手机时间
sleep_quality = 8 - 1.2 * stress + np.random.normal(0, 0.5, n)  # 睡眠质量评分

# 我们只观测到 X 和 Y —— 不知道 Z
df_omit = pd.DataFrame({'phone_hours': phone_hours, 'sleep_quality': sleep_quality})

# 画图：有遗漏变量 vs 控制后
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(df_omit['phone_hours'], df_omit['sleep_quality'], alpha=0.6,
           color=C_MAIN, edgecolors='white', linewidth=0.5, s=50)

# 不加控制（有偏）
X1 = sm.add_constant(df_omit['phone_hours'])
m1 = sm.OLS(df_omit['sleep_quality'], X1).fit()
xp = np.linspace(2, 7, 50)
yp1 = m1.params['const'] + m1.params['phone_hours'] * xp
ax.plot(xp, yp1, color=C_RED, linewidth=2.5, label=f'不考虑压力: β = {m1.params["phone_hours"]:.2f}')

# 控制压力后（真实效应）
# 用偏回归模拟：先分别去除压力影响
res_phone = sm.OLS(df_omit['phone_hours'], sm.add_constant(stress)).fit().resid
res_sleep = sm.OLS(df_omit['sleep_quality'], sm.add_constant(stress)).fit().resid
m2 = sm.OLS(res_sleep, sm.add_constant(res_phone)).fit()
true_beta = m2.params.iloc[1]  # 第二个参数 = 去中心化后的phone_hours系数
print(f"\n遗漏压力时: β(phone) = {m1.params['phone_hours']:.2f}")
print(f"控制压力后: β(phone) = {true_beta:.2f} (真实应为 0)")

ax.set_xlabel('每天刷手机时间（小时）', fontsize=12)
ax.set_ylabel('睡眠质量评分（1-10）', fontsize=12)
ax.set_title('图3：遗漏变量的魔力 — 错误的"负相关"', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.25)

# 注解
ax.text(2.2, 9.5,
        f'⚠️ 遗漏"压力"变量后\n刷手机→睡眠差的效应估计为 {m1.params["phone_hours"]:.1f}\n实际控制压力后效应≈0',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEDE1', alpha=0.9))

plt.tight_layout()
plt.savefig(r'C:\Users\qumin\Projects\zhihu-econometrics\images\003-omitted-variable.png',
            dpi=200, bbox_inches='tight')
print("✅ 遗漏变量效应图已保存")

print(f"""
╔══════════════════════════════════════════╗
║  遗漏变量效应演示                        ║
╠══════════════════════════════════════════╣
║  遗漏"压力"时:   β = {m1.params['phone_hours']:.2f}  (虚假显著)  ║
║  控制"压力"后:   β = {true_beta:.2f}  (真实为零)    ║
╚══════════════════════════════════════════╝
""")
