"""
知乎计量经济学系列 — 第 24 篇配套代码
主题：固定效应 vs 随机效应——到底该选谁？
工具：linearmodels (PanelOLS, RandomEffects), statsmodels
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels import PanelOLS, RandomEffects
from scipy import stats

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

# ========== 1. 模拟面板数据 ==========
# 设定：N=100 个员工，T=4 期（季度）
# 生成"个体效应"（每个员工的内在能力，不随时间变化）
# 关键：个体效应与核心解释变量（培训时长）相关 → 使 RE 不一致

N = 100
T = 4

# 个体效应 μ_i：每个员工的固有生产力
# 假设 μ_i 与培训投入相关（更努力的员工同时培训更多）
mu = np.random.normal(0, 1, N)

# 每个员工的培训偏好（与 μ_i 正相关）
train_alpha = 2 + mu + np.random.normal(0, 0.5, N)

# 生成面板数据
df_list = []
for i in range(N):
    for t in range(T):
        # 培训时长：受个体偏好和随机波动影响
        training = train_alpha[i] + np.random.normal(0, 0.5)

        # 工作年限（随时间增加）
        tenure = 1 + t + np.random.uniform(-0.3, 0.3)

        # 绩效产出：个体效应 + 培训效应 + 年限效应 + 噪声
        performance = (
            5                           # 截距
            + mu[i]                     # 个体效应（不随时间变）
            + 0.8 * training            # 培训的真实因果效应
            + 0.3 * tenure              # 工作年限的贡献
            + np.random.normal(0, 0.6)  # 随机误差
        )

        df_list.append({
            'employee': f'EMP_{i:03d}',
            'time': f'Q{t+1}',
            't': t,
            'training': training,
            'tenure': tenure,
            'performance': performance,
            'mu': mu[i],
        })

df = pd.DataFrame(df_list)

# 验证：个体效应与培训时长确实相关
print(f"个体效应 μ 与培训时长的相关系数: {df.groupby('employee')['mu'].first().corr(df.groupby('employee')['training'].mean()):.3f}")
print("→ 个体效应与解释变量相关 → RE 应不一致，FE 应一致\n")

# 设置面板索引
df = df.set_index(['employee', 't'])

# 自变量（不含截距——PanelOLS 会自行处理）
X = df[['training', 'tenure']]
y = df['performance']

# ========== 2. 固定效应估计 (PanelOLS) ==========
print("=" * 55)
print("固定效应 (FE) —— PanelOLS with entity effects")
print("=" * 55)
fe_model = PanelOLS(y, X, entity_effects=True)
fe_result = fe_model.fit(cov_type='robust')
print(fe_result)

fe_coefs = fe_result.params
fe_ci = fe_result.conf_int(level=0.95)

# ========== 3. 随机效应估计 (RandomEffects) ==========
print("\n" + "=" * 55)
print("随机效应 (RE) —— RandomEffects")
print("=" * 55)
re_model = RandomEffects(y, X)
re_result = re_model.fit(cov_type='robust')
print(re_result)

re_coefs = re_result.params
re_ci = re_result.conf_int(level=0.95)

# ========== 4. Hausman 检验 ==========
print("\n" + "=" * 55)
print("Hausman 检验")
print("=" * 55)

# 手动实现 Hausman 检验
# H = (b_FE - b_RE)' [Var(b_FE) - Var(b_RE)]⁻¹ (b_FE - b_RE)
# 共同变量
common_vars = [v for v in fe_coefs.index if v in re_coefs.index]

b_fe = fe_coefs[common_vars]
b_re = re_coefs[common_vars]
var_fe = fe_result.cov.loc[common_vars, common_vars]
var_re = re_result.cov.loc[common_vars, common_vars]

diff = b_fe - b_re
var_diff = var_fe - var_re

# 检查 var_diff 是否正定
try:
    # 使用广义逆
    from numpy.linalg import pinv
    var_diff_inv = pinv(var_diff)
    hausman_stat = diff @ var_diff_inv @ diff
    # 自由度 = 共同变量数
    dof = len(common_vars)
    p_value = 1 - stats.chi2.cdf(hausman_stat, dof)

    print(f"Hausman 统计量: χ²({dof}) = {hausman_stat:.4f}")
    print(f"p 值:           {p_value:.6f}")
    if p_value < 0.05:
        print("✅ p < 0.05 → 拒绝 RE 一致的原假设 → 应使用 FE")
    else:
        print("❌ p ≥ 0.05 → 不能拒绝 H₀ → RE 和 FE 无显著差异，RE 更有效率")
except Exception as e:
    print(f"Hausman 检验计算出错: {e}")
    print("使用 linearmodels 内置 Hausman 检验...")

# 尝试使用 linearmodels 的 Hausman 检验
try:
    from linearmodels.panel.model import compare
    hausman = compare({"FE": fe_result, "RE": re_result})
    print("\n模型对比（含 Hausman 检验）：")
    print(hausman)
except Exception:
    pass

print("\n" + "=" * 55)

# ========== 5. 图 1：FE vs RE 系数对比（含置信区间） ==========
fig, ax = plt.subplots(figsize=(10, 6))

variables = common_vars
x_pos = np.arange(len(variables))
width = 0.3

fe_y = [b_fe[v] for v in variables]
re_y = [b_re[v] for v in variables]
fe_err = [[fe_y[i] - fe_ci.loc[v, 'lower'] for i, v in enumerate(variables)],
          [fe_ci.loc[v, 'upper'] - fe_y[i] for i, v in enumerate(variables)]]
re_err = [[re_y[i] - re_ci.loc[v, 'lower'] for i, v in enumerate(variables)],
          [re_ci.loc[v, 'upper'] - re_y[i] for i, v in enumerate(variables)]]

bars1 = ax.bar(x_pos - width/2, fe_y, width, yerr=fe_err,
               color=C_MAIN, alpha=0.8, capsize=4, label='固定效应 (FE)',
               edgecolor='white', linewidth=0.5, error_kw={'linewidth': 1.5})
bars2 = ax.bar(x_pos + width/2, re_y, width, yerr=re_err,
               color=C_ACCENT, alpha=0.8, capsize=4, label='随机效应 (RE)',
               edgecolor='white', linewidth=0.5, error_kw={'linewidth': 1.5})

# 标注真实值
true_coefs = {'training': 0.8, 'tenure': 0.3}
for i, v in enumerate(variables):
    ax.axhline(y=true_coefs.get(v, 0), xmin=(i-0.3)/len(variables),
               xmax=(i+0.3)/len(variables), color=C_RED, linewidth=2,
               linestyle='--', alpha=0.7)

# 添加图例说明真实值
ax.plot([], [], color=C_RED, linestyle='--', linewidth=2,
        label='真实因果效应', alpha=0.7)

ax.set_xticks(x_pos)
variable_labels = {'training': '培训时长', 'tenure': '工作年限'}
ax.set_xticklabels([variable_labels.get(v, v) for v in variables], fontsize=12)
ax.set_ylabel('系数估计值', fontsize=12)
ax.set_title('图 1：固定效应 vs 随机效应——系数估计对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(axis='y', alpha=0.2)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/024-fe-re-comparison.png',
            dpi=200, bbox_inches='tight')
print("✅ 图 1 已保存：FE vs RE 系数对比图")

# ========== 6. 图 2：Hausman 检验可视化 ==========
# 模拟不同相关性强度下 FE 与 RE 的估计差异
corr_levels = np.linspace(0, 0.9, 10)
fe_bias = np.zeros(10)
re_bias = np.zeros(10)

for idx, corr in enumerate(corr_levels):
    # 重新模拟数据，控制个体效应与培训的相关系数
    mu_corr = np.random.normal(0, 1, N)
    train_corr = 2 + corr * mu_corr + np.random.normal(0, np.sqrt(1 - corr**2), N)

    df_c = []
    for i in range(N):
        for t in range(T):
            training_c = train_corr[i] + np.random.normal(0, 0.5)
            tenure_c = 1 + t + np.random.uniform(-0.3, 0.3)
            performance_c = 5 + mu_corr[i] + 0.8 * training_c + 0.3 * tenure_c + np.random.normal(0, 0.6)
            df_c.append({
                'employee': f'EMP_{i:03d}', 't': t,
                'training': training_c, 'tenure': tenure_c,
                'performance': performance_c,
            })

    df_c = pd.DataFrame(df_c).set_index(['employee', 't'])
    X_c = df_c[['training', 'tenure']]
    y_c = df_c['performance']

    try:
        fe_c = PanelOLS(y_c, X_c, entity_effects=True).fit(cov_type='robust')
        re_c = RandomEffects(y_c, X_c).fit(cov_type='robust')

        # 用 training 系数的偏差度量
        fe_bias[idx] = abs(fe_c.params['training'] - 0.8)
        re_bias[idx] = abs(re_c.params['training'] - 0.8)
    except Exception:
        fe_bias[idx] = np.nan
        re_bias[idx] = np.nan

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(corr_levels, fe_bias, 'o-', color=C_MAIN, linewidth=2.5,
         markersize=8, label='FE 偏差')
ax1.plot(corr_levels, re_bias, 's--', color=C_ACCENT, linewidth=2.5,
         markersize=8, label='RE 偏差')
ax1.fill_between(corr_levels, 0, re_bias, color=C_ACCENT, alpha=0.08)

# 标注 Hausman 检验所在位置（我们模拟的 ρ≈0.75）
sim_corr = df.groupby('employee')['mu'].first().corr(df.groupby('employee')['training'].mean())
ax1.axvline(x=sim_corr, color=C_RED, linestyle=':', linewidth=2, alpha=0.7)
ax1.annotate(f'模拟数据的相关系数\nρ ≈ {sim_corr:.2f}',
             xy=(sim_corr, 0.08), xytext=(sim_corr + 0.08, 0.15),
             fontsize=10, color=C_RED, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.5))

ax1.set_xlabel('个体效应与解释变量的相关系数 ρ', fontsize=12)
ax1.set_ylabel('系数估计的绝对偏差 |β̂ - β_true|', fontsize=12)
ax1.set_title('图 2：Hausman 检验直觉——相关性越强，RE 偏差越大', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/024-hausman-demo.png',
            dpi=200, bbox_inches='tight')
print("✅ 图 2 已保存：Hausman 检验可视化")

# ========== 7. 图 3：个体固定效应分布 ==========
# 从 FE 模型中提取估计的个体效应
fe_full = PanelOLS(y, X, entity_effects=True).fit()
# 个体效应存储在 params 中（以 employee 为名）
fe_entity_effects = fe_full.estimated_effects

fig, ax = plt.subplots(figsize=(10, 6))

# 直方图 + 核密度估计
fe_entity_effects_vals = fe_entity_effects.values.flatten()
ax.hist(fe_entity_effects_vals, bins=20, color=C_MAIN, alpha=0.6,
        edgecolor='white', linewidth=0.6, density=True, label='FE 估计的个体效应')

# 叠加核密度曲线
from scipy.stats import gaussian_kde
kde = gaussian_kde(fe_entity_effects_vals)
x_kde = np.linspace(fe_entity_effects_vals.min(), fe_entity_effects_vals.max(), 200)
ax.plot(x_kde, kde(x_kde), color=C_RED, linewidth=2.5, label='核密度估计')

# 标注真实的个体效应分布
ax.axvline(x=mu.mean(), color=C_GREEN, linestyle='--', linewidth=2, alpha=0.7)
ax.annotate(f'真实 μ 均值: {mu.mean():.2f}',
            xy=(mu.mean(), 0.40), fontsize=10, color=C_GREEN,
            fontweight='bold', ha='center')

ax.set_xlabel('个体效应估计值', fontsize=12)
ax.set_ylabel('密度', fontsize=12)
ax.set_title('图 3：估计的个体固定效应分布', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/024-individual-effects.png',
            dpi=200, bbox_inches='tight')
print("✅ 图 3 已保存：个体效应分布图")

# ========== 8. 汇总输出 ==========
print("\n" + "=" * 55)
print("关键结果汇总")
print("=" * 55)
print(f"""
真实因果效应: 培训时长 × 0.8, 工作年限 × 0.3

FE 估计:
  培训时长: {fe_coefs['training']:.3f} (95% CI: [{fe_ci.loc['training', 'lower']:.3f}, {fe_ci.loc['training', 'upper']:.3f}])
  工作年限: {fe_coefs['tenure']:.3f} (95% CI: [{fe_ci.loc['tenure', 'lower']:.3f}, {fe_ci.loc['tenure', 'upper']:.3f}])

RE 估计:
  培训时长: {re_coefs['training']:.3f} (95% CI: [{re_ci.loc['training', 'lower']:.3f}, {re_ci.loc['training', 'upper']:.3f}])
  工作年限: {re_coefs['tenure']:.3f} (95% CI: [{re_ci.loc['tenure', 'lower']:.3f}, {re_ci.loc['tenure', 'upper']:.3f}])

培训时长系数差异 (RE - FE): {re_coefs['training'] - fe_coefs['training']:.3f}
""")

print("🎉 第 24 篇所有图表已生成完毕！")
