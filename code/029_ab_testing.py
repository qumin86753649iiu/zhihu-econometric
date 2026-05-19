"""
知乎计量经济学系列 — 第 29 篇配套代码
主题：A/B 测试背后的计量逻辑
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.power import TTestIndPower

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

np.random.seed(42)

# ============================================================
# 图 1：Treatment vs Control 分布对比（KDE + t-test）
# ============================================================
print("=" * 50)
print("图 1：处理组 vs 对照组分布对比")
print("=" * 50)

n_ab = 500
# 对照组：均值为 50，标准差为 10
control = np.random.normal(50, 10, n_ab)
# 处理组：均值为 54（提升 8%），标准差为 10
treatment = np.random.normal(54, 10, n_ab)

t_stat, p_val = stats.ttest_ind(treatment, control)
effect_size = (treatment.mean() - control.mean()) / control.std()

print(f"对照组均值: {control.mean():.2f}，标准差: {control.std():.2f}")
print(f"处理组均值: {treatment.mean():.2f}，标准差: {treatment.std():.2f}")
print(f"均值差异: {treatment.mean() - control.mean():.2f}")
print(f"t 统计量: {t_stat:.4f}")
print(f"p 值: {p_val:.6f}")
print(f"Cohen's d (效应量): {effect_size:.3f}")

fig, ax = plt.subplots(figsize=(10, 6))

# KDE 曲线
sns.kdeplot(control, fill=True, color=C_GRAY, alpha=0.35,
            linewidth=2.5, label=f'对照组 (均值={control.mean():.1f})', ax=ax)
sns.kdeplot(treatment, fill=True, color=C_MAIN, alpha=0.35,
            linewidth=2.5, label=f'处理组 (均值={treatment.mean():.1f})', ax=ax)

# 标出均值位置
ax.axvline(control.mean(), color=C_GRAY, linestyle='--', linewidth=1.5, alpha=0.7)
ax.axvline(treatment.mean(), color=C_MAIN, linestyle='--', linewidth=1.5, alpha=0.7)

# 标注均值差异
mid_y = ax.get_ylim()[1] * 0.85
ax.annotate('', xy=(control.mean(), mid_y), xytext=(treatment.mean(), mid_y),
            arrowprops=dict(arrowstyle='<->', color=C_RED, lw=2.5))
ax.text((control.mean() + treatment.mean()) / 2, mid_y * 1.05,
         f'Δ = {treatment.mean() - control.mean():.1f}\np = {p_val:.4f}',
         ha='center', fontsize=12, fontweight='bold', color=C_RED)

ax.set_xlabel('转化率 / 指标值', fontsize=12)
ax.set_ylabel('密度', fontsize=12)
ax.set_title('图 1：A/B 测试——处理组 vs 对照组分发对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/029-ab-distribution.png', dpi=200, bbox_inches='tight')
print("✅ 图 1 已保存：029-ab-distribution.png\n")

# ============================================================
# 图 2：Simpson's Paradox 可视化
# ============================================================
print("=" * 50)
print("图 2：辛普森悖论可视化")
print("=" * 50)

np.random.seed(123)

# 构造三组存在辛普森悖论的数据
# 整体趋势：处理组均值 < 对照组均值（悖论）
# 但分层后：每个子组内部处理组均值 > 对照组均值
# 关键：不同组有不同的样本量（分流不均）

groups = ['年轻用户', '中年用户', '老年用户']
group_offsets = [30, 50, 70]  # 不同用户群的基础水平不同

# 模拟分流不均衡：
# 低基础组（年轻用户）有更多处理组用户
# 高基础组（老年用户）有更多对照组用户
# 这会拉低处理组的整体均值
sample_sizes_ctrl = [20, 50, 70]    # 对照组中：年轻少，老年多
sample_sizes_trt = [70, 50, 20]     # 处理组中：年轻多，老年少

all_control = []
all_treatment = []

for i, (gname, offset) in enumerate(zip(groups, group_offsets)):
    n_c = sample_sizes_ctrl[i]
    n_t = sample_sizes_trt[i]
    # 对照组：在基础水平附近
    c = np.random.normal(offset + 2, 5, n_c)
    # 处理组：在每个子组中效果略好于对照组（+3）
    t = np.random.normal(offset + 5, 5, n_t)
    all_control.append(c)
    all_treatment.append(t)

control_all = np.concatenate(all_control)
treatment_all = np.concatenate(all_treatment)
grand_mean_ctrl = control_all.mean()
grand_mean_trt = treatment_all.mean()

print(f"整体 - 对照组均值: {grand_mean_ctrl:.2f}")
print(f"整体 - 处理组均值: {grand_mean_trt:.2f}")
print(f"整体差异: {grand_mean_trt - grand_mean_ctrl:.2f}（处理组反而低！这是悖论）")

for i, gname in enumerate(groups):
    print(f"  {gname}: 对照组={all_control[i].mean():.2f}, "
          f"处理组={all_treatment[i].mean():.2f}, "
          f"差异={all_treatment[i].mean() - all_control[i].mean():+.2f}（处理组更高）")

fig, ax = plt.subplots(figsize=(12, 7))

colors_group = [C_MAIN, C_ACCENT, C_GREEN]

# 画每一组的散点和回归线
for i, (gname, offset) in enumerate(zip(groups, group_offsets)):
    n_c = sample_sizes_ctrl[i]
    n_t = sample_sizes_trt[i]
    x_ctrl = np.random.uniform(i - 0.2, i + 0.2, n_c)
    x_trt = np.random.uniform(i + 0.8 - 0.2, i + 0.8 + 0.2, n_t)

    # 对照组散点
    ax.scatter(x_ctrl, all_control[i], alpha=0.5, s=25,
               color=colors_group[i], edgecolors='white', linewidth=0.3,
               marker='o', label=f'{gname}-对照组' if i == 0 else '')
    # 处理组散点
    ax.scatter(x_trt, all_treatment[i], alpha=0.5, s=25,
               color=colors_group[i], edgecolors='white', linewidth=0.3,
               marker='^', label=f'{gname}-处理组' if i == 0 else '')

    # 组内均值连线
    mean_c = all_control[i].mean()
    mean_t = all_treatment[i].mean()
    ax.plot([i, i + 0.8], [mean_c, mean_t],
            color=colors_group[i], linewidth=2.5, linestyle='-', alpha=0.8)
    # 均值点
    ax.scatter([i], [mean_c], color=colors_group[i], s=120,
               edgecolors='white', linewidth=2, zorder=5)
    ax.scatter([i + 0.8], [mean_t], color=colors_group[i], s=120,
               edgecolors='white', linewidth=2, zorder=5,
               marker='^')

# 整体均值（虚线）
ax.axhline(grand_mean_ctrl, color=C_GRAY, linestyle='--', linewidth=2,
           alpha=0.6, label=f'整体对照组均值 ({grand_mean_ctrl:.1f})')
ax.axhline(grand_mean_trt, color=C_RED, linestyle='--', linewidth=2,
           alpha=0.6, label=f'整体处理组均值 ({grand_mean_trt:.1f})')

# 标注悖论箭头
ymin, ymax = ax.get_ylim()
ax.annotate('辛普森悖论！\n每组内部: 处理组 ↑\n整体来看: 处理组 ↓',
            xy=(1.4, grand_mean_trt), xytext=(1.4, ymax * 0.92),
            fontsize=11, fontweight='bold', color=C_RED,
            ha='center',
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.5))

ax.set_xticks([i + 0.4 for i in range(3)])
ax.set_xticklabels(groups, fontsize=12)
ax.set_xlim(-0.5, 3.3)
ax.set_ylabel('转化率 / 指标值', fontsize=12)
ax.set_title('图 2：辛普森悖论——每组内部处理组更高，但整体处理组更低', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='lower left', ncol=2)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/029-simpson-paradox.png', dpi=200, bbox_inches='tight')
print("✅ 图 2 已保存：029-simpson-paradox.png\n")

# ============================================================
# 图 3：Power Analysis 曲线
# ============================================================
print("=" * 50)
print("图 3：统计功效分析")
print("=" * 50)

analysis = TTestIndPower()
effect_sizes = np.linspace(0.1, 1.0, 50)

fig, ax = plt.subplots(figsize=(10, 6))

sample_sizes = [50, 100, 200, 500]
colors_power = [C_PURPLE, C_MAIN, C_GREEN, C_ACCENT]

for n, color in zip(sample_sizes, colors_power):
    powers = analysis.solve_power(effect_size=effect_sizes,
                                  nobs1=n, ratio=1.0, alpha=0.05,
                                  alternative='two-sided')
    ax.plot(effect_sizes, powers, color=color, linewidth=2.5,
            label=f'n = {n}（每组）')

# 80% 功效参考线
ax.axhline(y=0.8, color=C_RED, linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(0.7, 0.81, '80% 功效（常用阈值）', fontsize=11, color=C_RED,
        fontweight='bold', ha='center')

# 标注最小可检测效应量
for n, color in zip(sample_sizes, colors_power):
    min_effect = analysis.solve_power(nobs1=n, ratio=1.0,
                                       alpha=0.05, power=0.8,
                                       alternative='two-sided')
    ax.axvline(x=min_effect, color=color, linestyle=':', linewidth=1.2, alpha=0.5)
    ax.text(min_effect + 0.01, 0.1, f'{min_effect:.2f}',
            fontsize=8, color=color, rotation=90)

ax.set_xlabel('效应量 (Cohen\'s d)', fontsize=12)
ax.set_ylabel('统计功效 (Power)', fontsize=12)
ax.set_title('图 3：不同样本量下的统计功效曲线 (α=0.05)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(alpha=0.2)
ax.set_ylim(0, 1.05)
ax.set_xlim(0.1, 1.0)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/029-power-analysis.png', dpi=200, bbox_inches='tight')
print("✅ 图 3 已保存：029-power-analysis.png\n")

print("🎉 第 29 篇所有图表已生成完毕！")
