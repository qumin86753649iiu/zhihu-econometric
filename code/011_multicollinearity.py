"""
知乎计量经济学系列 — 第 11 篇配套代码
主题：多重共线性 — 两个朋友太像了
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ========== 全局样式 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
C_MAIN = '#2E86AB'
C_ACCENT = '#F18F01'
C_RED = '#C73E1D'
C_GREEN = '#4C9F70'
C_PURPLE = '#A23B72'
C_GRAY = '#A0A0A0'
IMAGE_DIR = r'C:\Users\qumin\projects\zhihu-econometrics\images'

np.random.seed(42)
n = 200

# ========== 1. 生成数据：年龄 vs 工作年限 vs 工资 ==========
# 真实关系：工资主要由年龄（反映职级）和经验共同决定
# 但年龄和工作年限高度相关！

age = np.random.normal(38, 10, n)  # 年龄 20-60
age = np.clip(age, 22, 60)

# 工作经验和年龄高度相关：年龄越大，通常经验越丰富
# 但同一年龄的人，经验也可能不同（比如转行、读研等）
experience = (age - 22) * 0.85 + np.random.normal(0, 3, n)  # 核心：经验≈年龄×0.85 + 噪声
experience = np.clip(experience, 0, 40)

# 教育年限：作为第三个变量，与年龄和经验弱相关
education = np.random.choice([12, 14, 16, 18, 20], n, p=[0.1, 0.15, 0.4, 0.25, 0.1])

# 工资 = 5000 + 年龄×800 + 经验×1200 + 教育×1500 + 噪声
salary = (5000 + age * 800 + experience * 1200 + education * 1500
          + np.random.normal(0, 15000, n))

df = pd.DataFrame({
    'age': age,
    'experience': experience,
    'education': education,
    'salary': salary
})

print("=== 数据预览 ===")
print(df.head())
print(f"\n年龄 vs 工作年限 相关系数: {df[['age', 'experience']].corr().iloc[0, 1]:.3f}")

# ========== 图1：年龄 vs 工作经验散点图 ==========
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(df['age'], df['experience'],
                     c=df['salary'], cmap='viridis', alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('工资 (元)', fontsize=10)

# 拟合线
X_line = sm.add_constant(df['age'])
m_line = sm.OLS(df['experience'], X_line).fit()
xp = np.linspace(20, 62, 100)
yp = m_line.params.iloc[0] + m_line.params.iloc[1] * xp
ax.plot(xp, yp, color=C_RED, linewidth=2.5, linestyle='--',
        label=f'拟合线: 经验 = {m_line.params.iloc[1]:.2f} × 年龄 + {m_line.params.iloc[0]:.1f}')

r2 = m_line.rsquared
ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
        fontsize=12, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax.set_xlabel('年龄', fontsize=12)
ax.set_ylabel('工作年限', fontsize=12)
ax.set_title('图1：年龄 vs 工作经验 — 高度相关', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/011-salary-scatter.png', dpi=200, bbox_inches='tight')
print("✅ 011-salary-scatter.png 已保存")

# ========== 图2：相关系数热力图 ==========
corr_matrix = df[['age', 'experience', 'education', 'salary']].corr()

fig, ax = plt.subplots(figsize=(7, 6))
mask = None  # 显示全部三角
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap=cmap,
            vmin=-1, vmax=1, center=0, square=True,
            linewidths=0.8, cbar_kws={'shrink': 0.8},
            annot_kws={'fontsize': 11}, ax=ax)
ax.set_title('图2：变量间的相关系数矩阵', fontsize=14, fontweight='bold')
ax.set_xticklabels(['年龄', '工作年限', '教育年限', '工资'], fontsize=10)
ax.set_yticklabels(['年龄', '工作年限', '教育年限', '工资'], fontsize=10, rotation=0)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/011-correlation-heatmap.png', dpi=200, bbox_inches='tight')
print("✅ 011-correlation-heatmap.png 已保存")

# ========== 图3：VIF 条形图 ==========
# 计算 VIF 值
X_vif = df[['age', 'experience', 'education']]
X_vif = sm.add_constant(X_vif)

vif_data = pd.DataFrame()
vif_data['变量'] = ['年龄 (Age)', '工作年限 (Exp)', '教育年限 (Edu)']
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(1, X_vif.shape[1])]
vif_data['容忍度'] = 1.0 / vif_data['VIF']

print("\n=== VIF 诊断结果 ===")
print(vif_data.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5.5))
bar_colors = []
for v in vif_data['VIF']:
    if v > 10:
        bar_colors.append(C_RED)
    elif v > 5:
        bar_colors.append(C_ACCENT)
    else:
        bar_colors.append(C_GREEN)

bars = ax.bar(vif_data['变量'], vif_data['VIF'], color=bar_colors, edgecolor='white', linewidth=1.2, width=0.55)

# 阈值线
ax.axhline(y=10, color=C_RED, linewidth=2, linestyle='--', alpha=0.7, label='VIF=10（严重共线性阈值）')
ax.axhline(y=5, color=C_ACCENT, linewidth=2, linestyle=':', alpha=0.7, label='VIF=5（中度共线性阈值）')

# 在柱子上标注数值
for bar, v in zip(bars, vif_data['VIF']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f'{v:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

title_str = ('图3：VIF（方差膨胀因子）诊断\n'
             '年龄和工作年限的 VIF > 5，存在中等多重共线性')
ax.set_title(title_str, fontsize=13, fontweight='bold')
ax.set_ylabel('方差膨胀因子 (VIF)', fontsize=11)
ax.set_ylim(0, max(vif_data['VIF']) * 1.3)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/011-vif-bar.png', dpi=200, bbox_inches='tight')
print("✅ 011-vif-bar.png 已保存")

# ========== 图4：系数不稳定性演示 ==========
# 核心思想：对数据进行小扰动（如去掉少量样本），观察系数变化
# 高共线性下，系数的波动远大于低共线性

np.random.seed(42)
n_sim = 500  # 模拟次数

# 高共线性情景：age 和 experience 强相关
coef_age_high = []
coef_exp_high = []
coef_edu_high = []

# 低共线性情景：age 和 experience 弱相关（人为去相关）
coef_age_low = []
coef_exp_low = []

for i in range(n_sim):
    # 每次重新采样（bootstrap）
    idx = np.random.choice(n, n, replace=True)
    df_boot = df.iloc[idx]

    # --- 高共线性模型 ---
    X_high = sm.add_constant(df_boot[['age', 'experience', 'education']])
    m_high = sm.OLS(df_boot['salary'], X_high).fit()
    coef_age_high.append(m_high.params.iloc[1])
    coef_exp_high.append(m_high.params.iloc[2])
    coef_edu_high.append(m_high.params.iloc[3])

    # --- 低共线性模型：使用去相关的经验变量 ---
    # 构造一个与年龄正交的经验变量
    age_resid = df_boot['experience'] - m_line.predict(sm.add_constant(df_boot['age']))
    X_low = sm.add_constant(pd.DataFrame({
        'age': df_boot['age'],
        'experience_orth': age_resid,  # 正交化的经验（仅保留与年龄无关的部分）
        'education': df_boot['education']
    }))
    m_low = sm.OLS(df_boot['salary'], X_low).fit()
    coef_age_low.append(m_low.params.iloc[1])
    coef_exp_low.append(m_low.params.iloc[2])

coef_df = pd.DataFrame({
    'age_high': coef_age_high,
    'exp_high': coef_exp_high,
    'edu_high': coef_edu_high,
    'age_low': coef_age_low,
    'exp_low': coef_exp_low,
})

# 统计波动范围
print("\n=== 系数稳定性对比 ===")
for name, col in [('年龄(高共线)', 'age_high'), ('经验(高共线)', 'exp_high'),
                   ('教育(高共线)', 'edu_high'), ('年龄(低共线)', 'age_low'),
                   ('经验(低共线)', 'exp_low')]:
    print(f"  {name}: 均值={coef_df[col].mean():.1f}, "
          f"标准差={coef_df[col].std():.1f}, "
          f"范围=[{coef_df[col].min():.1f}, {coef_df[col].max():.1f}]")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 左图：高共线性下的系数分布
ax1 = axes[0]
ax1.hist(coef_age_high, bins=35, alpha=0.5, color=C_MAIN, label=f'年龄系数\n(均={np.mean(coef_age_high):.0f}, 标准差={np.std(coef_age_high):.0f})')
ax1.hist(coef_exp_high, bins=35, alpha=0.5, color=C_ACCENT, label=f'经验系数\n(均={np.mean(coef_exp_high):.0f}, 标准差={np.std(coef_exp_high):.0f})')
ax1.hist(coef_edu_high, bins=35, alpha=0.5, color=C_GREEN, label=f'教育系数\n(均={np.mean(coef_edu_high):.0f}, 标准差={np.std(coef_edu_high):.0f})')
ax1.axvline(x=0, color='gray', linewidth=1, linestyle='--')
ax1.set_xlabel('系数值', fontsize=11)
ax1.set_ylabel('频次 (500次Bootstrap)', fontsize=11)
ax1.set_title('高共线性: 年龄与经验高度相关', fontsize=13, fontweight='bold')
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(alpha=0.2)

# 右图：低共线性下的系数分布
ax2 = axes[1]
ax2.hist(coef_age_low, bins=35, alpha=0.5, color=C_MAIN, label=f'年龄系数(正交化后)\n(均={np.mean(coef_age_low):.0f}, 标准差={np.std(coef_age_low):.0f})')
ax2.hist(coef_exp_low, bins=35, alpha=0.5, color=C_PURPLE, label=f'经验残差系数\n(均={np.mean(coef_exp_low):.0f}, 标准差={np.std(coef_exp_low):.0f})')
ax2.axvline(x=0, color='gray', linewidth=1, linestyle='--')
ax2.set_xlabel('系数值', fontsize=11)
ax2.set_ylabel('频次 (500次Bootstrap)', fontsize=11)
ax2.set_title('低共线性: 经验正交化后', fontsize=13, fontweight='bold')
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(alpha=0.2)

plt.suptitle('图4：多重共线性导致系数不稳定', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/011-coefficient-instability.png', dpi=200, bbox_inches='tight')
print("✅ 011-coefficient-instability.png 已保存")

# ========== 5. 输出摘要 ==========
print(f"""
╔══════════════════════════════════════════════════════════════╗
║  多重共线性诊断报告                                          ║
╠══════════════════════════════════════════════════════════════╣
║  原始模型: salary = β₀ + β₁·age + β₂·experience + β₃·education ║
╠══════════════════════════════════════════════════════════════╣
║  相关系数 (age, experience) = {df[['age', 'experience']].corr().iloc[0, 1]:.3f}       ║
║                                                            ║
║  VIF 诊断:                                                  ║
║    年龄 (Age)       VIF = {vif_data.iloc[0, 1]:.2f}                              ║
║    工作年限 (Exp)   VIF = {vif_data.iloc[1, 1]:.2f}                              ║
║    教育年限 (Edu)   VIF = {vif_data.iloc[2, 1]:.2f}                              ║
║                                                            ║
║  系数稳定性 (500次Bootstrap 标准差):                           ║
║    年龄(高共线)    {np.std(coef_age_high):.0f}                                  ║
║    经验(高共线)    {np.std(coef_exp_high):.0f}                                  ║
║    年龄(低共线)    {np.std(coef_age_low):.0f}                                  ║
║    经验(低共线)    {np.std(coef_exp_low):.0f}                                  ║
╚══════════════════════════════════════════════════════════════╝
""")
