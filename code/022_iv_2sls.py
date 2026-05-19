"""
知乎计量经济学系列 — 第 22 篇配套代码
主题：工具变量——用距离来回答因果（2SLS）
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

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

# ========== 1. 模拟数据：距离 → 大学教育 → 收入 ==========
n = 1000

# —— 不可观测的混杂因素：能力/动力 ——
ability = np.random.normal(0, 1, n)

# —— 工具变量：家到最近大学的距离（对数） ——
distance = np.random.lognormal(mean=1.5, sigma=0.8, size=n)
log_distance = np.log(distance)
# 距离范围从几百米到几十公里
print(f"距离统计：均值={distance.mean():.1f}km, 中位数={np.median(distance):.1f}km, "
      f"范围=[{distance.min():.1f}, {distance.max():.1f}]km")

# —— 内生变量：大学教育年限 ——
# 距离越远，受教育年限越少（相关性）
# 能力越强，受教育年限越多
college = (
    14.0                         # 基准（高中毕业后约14年教育）
    - 0.9 * log_distance         # 距离的负效应
    + 0.8 * ability              # 能力的正向效应
    + np.random.normal(0, 0.6, n)
)
# 截断到合理范围 [9, 22]
college = np.clip(college, 9, 22)

# —— 结果变量：收入（万元/年） ——
# 真实因果效应：多一年大学教育，收入增加 0.5 万
# 能力也直接影响收入（混杂因素）
earnings = (
    2.0
    + 0.5 * college              # 真实因果效应 ← 这是我们想估计的
    + 0.6 * ability              # 混杂：能力高的人赚更多
    + np.random.normal(0, 1.0, n)
)
earnings = np.clip(earnings, 0, 40)

df = pd.DataFrame({
    'log_distance': log_distance,
    'distance': distance,
    'college': college,
    'ability': ability,
    'earnings': earnings
})

print(f"样本量: {n}")
print(f"大学教育年限: 均值={college.mean():.2f}, 标准差={college.std():.2f}")
print(f"年收入(万元): 均值={earnings.mean():.2f}, 标准差={earnings.std():.2f}")

# ========== 2. OLS：有偏估计（因为忽略了 ability） ==========
X_ols = sm.add_constant(df[['college']])
ols_model = sm.OLS(df['earnings'], X_ols).fit()

print(f"\n{'='*55}")
print("OLS 回归结果（有偏——忽略了能力等混杂因素）")
print(f"{'='*55}")
print(f"  教育回报率估计: {ols_model.params['college']:.4f}  (真实值: 0.5000)")
print(f"  标准误:         {ols_model.bse['college']:.4f}")
print(f"  95% CI:         [{ols_model.conf_int().loc['college', 0]:.4f}, "
      f"{ols_model.conf_int().loc['college', 1]:.4f}]")

bias_pct = (ols_model.params['college'] - 0.5) / 0.5 * 100
print(f"  偏差:           {bias_pct:+.1f}%")

# ========== 3. 第一阶段：log(distance) → college ==========
X_stage1 = sm.add_constant(df[['log_distance']])
stage1 = sm.OLS(df['college'], X_stage1).fit()
df['college_hat'] = stage1.fittedvalues  # 保存拟合值

print(f"\n{'='*55}")
print("第一阶段：log(距离) → 大学教育年限")
print(f"{'='*55}")
print(f"  log(距离)系数: {stage1.params['log_distance']:.4f}")
print(f"  t 值:         {stage1.tvalues['log_distance']:.2f}")
print(f"  R²:           {stage1.rsquared:.4f}")
print(f"  F 统计量:     {stage1.fvalue:.2f}")

# 弱工具变量检验：F > 10 是经验法则
f_stat = stage1.fvalue
print(f"  -> {'✅ 强工具变量 (F > 10)' if f_stat > 10 else '⚠️ 弱工具变量 (F < 10)'}")

# ========== 4. 第二阶段：college_hat → earnings ==========
X_stage2 = sm.add_constant(df[['college_hat']])
stage2 = sm.OLS(df['earnings'], X_stage2).fit()

print(f"\n{'='*55}")
print("第二阶段（手动）：预测大学教育 → 收入")
print(f"{'='*55}")
print(f"  IV 教育回报率估计: {stage2.params['college_hat']:.4f}  (真实值: 0.5000)")
print(f"  标准误:            {stage2.bse['college_hat']:.4f}")

iv_bias_pct = (stage2.params['college_hat'] - 0.5) / 0.5 * 100
print(f"  偏差:              {iv_bias_pct:+.1f}%")

# ========== 5. 正式 2SLS：用 linearmodels ==========
df['const'] = 1
iv_model = IV2SLS(
    dependent=df['earnings'],
    exog=df[['const']],
    endog=df[['college']],
    instruments=df[['log_distance']]
)
iv_result = iv_model.fit(cov_type='robust')

print(f"\n{'='*55}")
print("IV2SLS（正式——使用 linearmodels）")
print(f"{'='*55}")
print(iv_result.summary)

# 提取关键结果
iv_coef = iv_result.params['college']
iv_se = iv_result.std_errors['college']
iv_ci_low = iv_result.conf_int().loc['college', 'lower']
iv_ci_high = iv_result.conf_int().loc['college', 'upper']

print(f"\n  IV 教育回报率估计: {iv_coef:.4f}  (真实值: 0.5000)")
print(f"  稳健标准误:       {iv_se:.4f}")
print(f"  95% CI:           [{iv_ci_low:.4f}, {iv_ci_high:.4f}]")

iv_bias_pct2 = (iv_coef - 0.5) / 0.5 * 100
print(f"  偏差:             {iv_bias_pct2:+.1f}%")

# ========== 图 1：第一阶段——工具变量（距离）预测内生变量（大学教育） ==========
fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(df['log_distance'], df['college'], alpha=0.3, color=C_MAIN,
           edgecolors='white', linewidth=0.2, s=20, label='观测数据')

# 第一阶段回归线
x_line = np.linspace(df['log_distance'].min(), df['log_distance'].max(), 200)
y_line = stage1.params['const'] + stage1.params['log_distance'] * x_line
ax.plot(x_line, y_line, color=C_RED, linewidth=2.5,
        label=f'第一阶段: college = {stage1.params["const"]:.2f} '
              f'+ {stage1.params["log_distance"]:.2f} × log(距离)')

ax.annotate(f'F = {stage1.fvalue:.1f}\n(> 10 → 强工具变量)',
            xy=(0.05, 0.9), xycoords='axes fraction', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))

ax.set_xlabel('家到最近大学的距离（对数）', fontsize=12)
ax.set_ylabel('大学教育年限', fontsize=12)
ax.set_title('第一阶段：距离越远 → 受教育年限越少（相关性）', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower left')
ax.grid(alpha=0.15)

# 添加中文刻度说明——在底部加一条距离数值参考
sec_ax = ax.secondary_xaxis('bottom', functions=(np.exp, np.log))
sec_ax.set_xlabel('距离（公里）', fontsize=9, labelpad=8)
sec_ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/022-iv-first-stage.png', dpi=200, bbox_inches='tight')
print(f"\n✅ 图 1 已保存：022-iv-first-stage.png")

# ========== 图 2：第二阶段——预测的大学教育 → 收入 ==========
fig, ax = plt.subplots(figsize=(9, 6))

# 用第一阶段拟合值画散点
ax.scatter(df['college'], df['earnings'], alpha=0.15, color=C_GRAY,
           edgecolors='white', linewidth=0.2, s=15, label='原始数据（灰色）')
ax.scatter(df['college_hat'], df['earnings'], alpha=0.5, color=C_ACCENT,
           edgecolors='white', linewidth=0.3, s=25, label='使用预测的教育年限（橙色）')

# 第二阶段回归线
x_line2 = np.linspace(df['college_hat'].min(), df['college_hat'].max(), 200)
y_line2 = stage2.params['const'] + stage2.params['college_hat'] * x_line2
ax.plot(x_line2, y_line2, color=C_RED, linewidth=2.5,
        label=f'第二阶段: earnings = {stage2.params["const"]:.2f} '
              f'+ {stage2.params["college_hat"]:.2f} × collegê')

ax.set_xlabel('大学教育年限（橙色为预测值 collegê）', fontsize=12)
ax.set_ylabel('年收入（万元）', fontsize=12)
ax.set_title('第二阶段：仅用距离预测出的教育 → 对收入的影响（排他性）', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.15)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/022-iv-second-stage.png', dpi=200, bbox_inches='tight')
print(f"✅ 图 2 已保存：022-iv-second-stage.png")

# ========== 图 3：OLS vs IV 系数对比（带置信区间） ==========
fig, ax = plt.subplots(figsize=(8, 5))

estimates = {
    'OLS\n（有偏）': {
        'coef': ols_model.params['college'],
        'ci_low': ols_model.conf_int().loc['college', 0],
        'ci_high': ols_model.conf_int().loc['college', 1]
    },
    'IV (2SLS)\n（无偏）': {
        'coef': iv_coef,
        'ci_low': iv_ci_low,
        'ci_high': iv_ci_high
    }
}

labels = list(estimates.keys())
coefs = [estimates[k]['coef'] for k in labels]
err_low = [estimates[k]['coef'] - estimates[k]['ci_low'] for k in labels]
err_high = [estimates[k]['ci_high'] - estimates[k]['coef'] for k in labels]

colors = [C_GRAY, C_MAIN]
x_pos = [0, 1]

for i, (x, c, cl, ch) in enumerate(zip(x_pos, coefs, err_low, err_high)):
    ax.errorbar(x, c, yerr=[[cl], [ch]], fmt='o', color=colors[i],
                capsize=6, capthick=2, markersize=12, linewidth=2.5)

# 真实值参考线
ax.axhline(y=0.5, color=C_RED, linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'真实因果效应 = 0.50')

# 标注数值
for i, (x, c, cl, ch) in enumerate(zip(x_pos, coefs, err_low, err_high)):
    ax.annotate(f'{c:.3f}\n[{c-cl:.3f}, {c+ch:.3f}]',
                xy=(x, c), xytext=(x, c + 0.12 if i == 0 else c + 0.08),
                ha='center', fontsize=10, fontweight='bold',
                color=colors[i],
                arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.2))

ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('教育回报率（每多一年大学，收入增加多少万元）', fontsize=11)
ax.set_title('OLS vs IV（2SLS）：工具变量如何消除混杂偏差', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(axis='y', alpha=0.2)
ax.set_xlim(-0.6, 1.6)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/022-iv-comparison.png', dpi=200, bbox_inches='tight')
print(f"✅ 图 3 已保存：022-iv-comparison.png")

# ========== 6. 输出汇总表 ==========
print(f"\n{'='*60}")
print(f"{'方法':>12} {'估计值':>10} {'真实值':>10} {'偏差%':>10} {'95% CI':>22}")
print(f"{'-'*60}")
print(f"{'OLS':>12} {ols_model.params['college']:>10.4f} {'0.5000':>10} {bias_pct:>+9.1f}%  "
      f"[{ols_model.conf_int().loc['college', 0]:.3f}, "
      f"{ols_model.conf_int().loc['college', 1]:.3f}]")
print(f"{'IV (2SLS)':>12} {iv_coef:>10.4f} {'0.5000':>10} {iv_bias_pct2:>+9.1f}%  "
      f"[{iv_ci_low:.3f}, {iv_ci_high:.3f}]")
print(f"{'='*60}")

print(f"\n🎉 第 22 篇所有图表已生成完毕！")
print(f"   总结：OLS 高估了教育回报率 (因为有 ability 混杂)，")
print(f"   而 IV 通过距离这个工具变量成功恢复了真实的因果效应。")
