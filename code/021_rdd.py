"""
知乎计量经济学系列 — 第 21 篇配套代码
主题：断点回归 RDD——多一点就进了名校？
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
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

# ===== 固定参数 =====
np.random.seed(42)
N = 10000
CUTOFF = 600
TRUE_EFFECT = 5.0  # 因果效应：千元/月

# ========== 1. 模拟数据 ==========
print("=" * 55)
print("第 21 篇：断点回归 RDD — 数据和图表生成")
print("=" * 55)

score = np.random.uniform(300, 750, N)
admitted = (score >= CUTOFF).astype(float)

# 基础收入：与成绩线性相关
income_base = 5 + 0.03 * (score - CUTOFF)

# 因果效应跳跃
epsilon = np.random.normal(0, 2.5, N)
income = income_base + TRUE_EFFECT * admitted + epsilon

df = pd.DataFrame({'score': score, 'admitted': admitted, 'income': income})

print(f"样本量: {N}")
print(f"录取比例: {admitted.mean():.2%}")
print(f"真实因果效应: {TRUE_EFFECT} 千元/月")
print()

# ========== 2. 图 1 — 断点回归核心散点图 ==========
h_main = 50  # 用于主图的带宽
sub = df[(df['score'] >= CUTOFF - h_main) & (df['score'] <= CUTOFF + h_main)].copy()
left = sub[sub['score'] < CUTOFF]
right = sub[sub['score'] >= CUTOFF]

fig, ax = plt.subplots(figsize=(12, 7))

# 散点
ax.scatter(left['score'], left['income'], alpha=0.25,
           color=C_GREEN, s=12, edgecolors='white', linewidth=0.2, label='未录取（对照组）')
ax.scatter(right['score'], right['income'], alpha=0.25,
           color=C_RED, s=12, edgecolors='white', linewidth=0.2, label='录取（处理组）')

# 线性拟合
for subset, color, label in [
    (left, C_MAIN, '左拟合线'),
    (right, C_PURPLE, '右拟合线'),
]:
    X_loc = sm.add_constant(subset['score'] - CUTOFF)
    m_loc = sm.OLS(subset['income'], X_loc).fit()
    xp = np.linspace((subset['score'] - CUTOFF).min(),
                     (subset['score'] - CUTOFF).max(), 200)
    yp = m_loc.params.iloc[0] + m_loc.params.iloc[1] * xp
    ax.plot(xp + CUTOFF, yp, color=color, linewidth=3, label=label)

    # 在断点处画预测点
    pred_at_cutoff = m_loc.params.iloc[0]
    ax.scatter(CUTOFF, pred_at_cutoff, color=color, s=120, zorder=10,
               edgecolors='white', linewidth=2)

# 标注跳跃
ax.annotate(f'因果效应 τ ≈ {TRUE_EFFECT} 千元/月',
            xy=(CUTOFF + 2, 11.5),
            xytext=(CUTOFF + 45, 14.5),
            fontsize=13, color=C_RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=2.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# 断点竖线
ax.axvline(x=CUTOFF, color=C_GRAY, linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(CUTOFF + 2, ax.get_ylim()[0] + 1, f'分数线 = {CUTOFF}',
        fontsize=11, color=C_GRAY, fontstyle='italic')

ax.set_xlabel('高考成绩', fontsize=14)
ax.set_ylabel('未来月收入（千元）', fontsize=14)
ax.set_title('图 1：断点回归——分数线附近的因果效应', fontsize=15, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.15)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/021-rdd-scatter.png', dpi=200, bbox_inches='tight')
print("✅ 图 1 已保存：021-rdd-scatter.png")

# ========== 3. 图 2 — 带宽敏感性分析 ==========
bandwidths = np.arange(5, 101, 5)
estimates = []
ci_lowers = []
ci_uppers = []

for h in bandwidths:
    sub_h = df[(df['score'] >= CUTOFF - h) & (df['score'] <= CUTOFF + h)]
    sub_l = sub_h[sub_h['score'] < CUTOFF]
    sub_r = sub_h[sub_h['score'] >= CUTOFF]

    if len(sub_l) < 3 or len(sub_r) < 3:
        estimates.append(np.nan)
        ci_lowers.append(np.nan)
        ci_uppers.append(np.nan)
        continue

    # 左侧回归
    X_l = sm.add_constant(sub_l['score'] - CUTOFF)
    m_l = sm.OLS(sub_l['income'], X_l).fit()
    pred_l = m_l.params.iloc[0]

    # 右侧回归
    X_r = sm.add_constant(sub_r['score'] - CUTOFF)
    m_r = sm.OLS(sub_r['income'], X_r).fit()
    pred_r = m_r.params.iloc[0]

    tau = pred_r - pred_l
    estimates.append(tau)

    # 简化的标准误
    se = np.sqrt(m_l.bse.iloc[0]**2 + m_r.bse.iloc[0]**2)
    ci_lowers.append(tau - 1.96 * se)
    ci_uppers.append(tau + 1.96 * se)

estimates = np.array(estimates)
ci_lowers = np.array(ci_lowers)
ci_uppers = np.array(ci_uppers)

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(bandwidths, estimates, color=C_MAIN, linewidth=2.5, marker='o',
        markersize=6, label='RDD 估计值')
ax.fill_between(bandwidths, ci_lowers, ci_uppers,
                color=C_MAIN, alpha=0.15, label='95% 置信区间')
ax.axhline(y=TRUE_EFFECT, color=C_RED, linestyle='--', linewidth=1.5,
           label=f'真实效应 = {TRUE_EFFECT}')

# 标注带宽过小和过大的区域
ax.axvspan(5, 15, alpha=0.08, color=C_RED, label='带宽过小（高方差）')
ax.axvspan(75, 100, alpha=0.08, color=C_ACCENT, label='带宽过大（可能偏差）')

ax.set_xlabel('带宽（分数 ±h）', fontsize=13)
ax.set_ylabel('因果效应估计值', fontsize=13)
ax.set_title('图 2：带宽敏感性分析——不同带宽下的 RDD 估计', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/021-rdd-bandwidth.png', dpi=200, bbox_inches='tight')
print("✅ 图 2 已保存：021-rdd-bandwidth.png")

# ========== 4. 图 3 — 密度检验（McCrary 检验） ==========
fig, ax = plt.subplots(figsize=(10, 6))

bins = np.linspace(CUTOFF - 50, CUTOFF + 50, 35)
# 左侧直方图
ax.hist(left['score'], bins=bins[bins < CUTOFF],
        alpha=0.7, color=C_GREEN, label='左侧（未录取）',
        density=True, edgecolor='white')
# 右侧直方图
ax.hist(right['score'], bins=bins[bins >= CUTOFF],
        alpha=0.7, color=C_RED, label='右侧（录取）',
        density=True, edgecolor='white')

ax.axvline(x=CUTOFF, color=C_GRAY, linestyle='--', linewidth=1.5)

# 添加密度拟合曲线（核密度估计）
from scipy.stats import gaussian_kde
for subset, color, label in [
    (left['score'].values, C_MAIN, '左侧 KDE'),
    (right['score'].values, C_PURPLE, '右侧 KDE'),
]:
    if len(subset) > 5:
        kde = gaussian_kde(subset)
        x_grid = np.linspace(subset.min(), subset.max(), 200)
        ax.plot(x_grid, kde(x_grid), color=color, linewidth=2.5, label=label)

ax.set_xlabel('高考成绩', fontsize=13)
ax.set_ylabel('概率密度', fontsize=13)
ax.set_title('图 3：McCrary 密度检验——断点两侧密度是否连续？', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/021-rdd-density.png', dpi=200, bbox_inches='tight')
print("✅ 图 3 已保存：021-rdd-density.png")

# ========== 5. 二次模型对比 ==========
print("\n" + "=" * 55)
print("模型对比：线性 vs 二次")
print("=" * 55)

h_opt = 50  # 使用 h=50 作为"最优"带宽

sub_opt = df[(df['score'] >= CUTOFF - h_opt) & (df['score'] <= CUTOFF + h_opt)].copy()
sub_opt['score_c'] = sub_opt['score'] - CUTOFF
sub_opt['D'] = sub_opt['admitted']

# ---- 线性模型 ----
X_linear = sm.add_constant(sub_opt[['score_c', 'D']])
# 交互项
X_linear['D_x_score_c'] = sub_opt['D'] * sub_opt['score_c']

m_linear = sm.OLS(sub_opt['income'], X_linear).fit()
tau_linear = m_linear.params['D']

print(f"\n线性模型（带宽 h={h_opt}）：")
print(f"  τ (因果效应) = {tau_linear:.3f}   (真实值 = {TRUE_EFFECT})")
print(f"  标准误 = {m_linear.bse['D']:.3f}")
print(f"  R² = {m_linear.rsquared:.3f}")

# ---- 二次模型 ----
X_quad = X_linear.copy()
X_quad['score_c2'] = sub_opt['score_c'] ** 2
X_quad['D_x_score_c2'] = sub_opt['D'] * sub_opt['score_c'] ** 2

m_quad = sm.OLS(sub_opt['income'], X_quad).fit()
tau_quad = m_quad.params['D']

print(f"\n二次模型（带宽 h={h_opt}）：")
print(f"  τ (因果效应) = {tau_quad:.3f}   (真实值 = {TRUE_EFFECT})")
print(f"  标准误 = {m_quad.bse['D']:.3f}")
print(f"  R² = {m_quad.rsquared:.3f}")

print("\n" + "=" * 55)
print("带宽敏感性分析汇总：")
print("=" * 55)
print(f"{'带宽':>6} {'估计值':>8} {'95%CI下界':>10} {'95%CI上界':>10}")
print("-" * 40)
step = max(1, len(bandwidths) // 10)
for i in range(0, len(bandwidths), step):
    h = bandwidths[i]
    if not np.isnan(estimates[i]):
        print(f"{h:>6} {estimates[i]:>8.3f} {ci_lowers[i]:>10.3f} {ci_uppers[i]:>10.3f}")

print("\n🎉 第 21 篇所有图表已生成完毕！")
print(f"   生成文件：{IMAGE_DIR}/021-rdd-scatter.png")
print(f"   生成文件：{IMAGE_DIR}/021-rdd-bandwidth.png")
print(f"   生成文件：{IMAGE_DIR}/021-rdd-density.png")
