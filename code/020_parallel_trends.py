"""
知乎计量经济学系列 — 第 20 篇配套代码
主题：平行趋势——DID 的生命线（事件研究图 + 安慰剂检验）
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

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

# ========== 1. 模拟面板数据 ==========
print("=" * 50)
print("生成面板数据...")
np.random.seed(42)
N = 100          # 100 个个体
treat_time = 0   # 第 0 期开始处理
periods = list(range(-5, 5))  # -5 到 4 共 10 期

units = list(range(N))
data = []
for u in units:
    treat = 1 if u < 50 else 0
    mu_i = np.random.normal(0, 0.5)  # 个体固定效应
    for t in periods:
        lambda_t = np.random.normal(0, 0.2)  # 时间固定效应
        trend = 0.3 * t  # 共同趋势
        # 政策效应：处理组在政策后的效应，随时间递增
        effect = 0 if t < treat_time else (1.0 + 0.3 * (t - treat_time))
        y = 10 + trend + mu_i + lambda_t + effect * treat + np.random.normal(0, 0.5)
        data.append({'unit': u, 'time': t, 'treat': treat,
                     'y': y, 'post': 1 if t >= treat_time else 0})

df = pd.DataFrame(data)
print(f"  面板数据形状: {df.shape}")
print(f"  个体数: {df['unit'].nunique()}, 时期数: {df['time'].nunique()}")

# ========== 2. 事件研究：构建交互项 ==========
print("\n构建事件研究模型...")
for t in periods:
    df[f'time_{t}'] = (df['time'] == t).astype(int)

# 事件研究交互项 (treat × time dummy)，以 time_-1 为基准组
for t in periods:
    if t == -1:
        continue
    df[f'treat_x_time_{t}'] = df['treat'] * df[f'time_{t}']

# 构造回归变量
event_vars = [f'treat_x_time_{t}' for t in periods if t != -1]
time_fe_vars = [f'time_{t}' for t in periods if t != 0]
X_event = sm.add_constant(df[['treat'] + time_fe_vars + event_vars])
m_event = sm.OLS(df['y'], X_event).fit()

# 提取事件研究系数
event_coefs = {}
event_cis = {}
for t in periods:
    if t == -1:
        event_coefs[t] = 0.0
        event_cis[t] = (0.0, 0.0)
    else:
        vname = f'treat_x_time_{t}'
        coef = m_event.params[vname]
        se = m_event.bse[vname]
        event_coefs[t] = coef
        event_cis[t] = (coef - 1.96 * se, coef + 1.96 * se)

print("  事件研究系数（政策前各期应接近 0）:")
for t in sorted(event_coefs.keys()):
    ci = event_cis[t]
    sig = '*' if (ci[0] > 0 or ci[1] < 0) else ' '
    print(f"    t={t:+d}: {event_coefs[t]:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]{sig}")

# ===== 图 1：事件研究图 =====
fig, ax = plt.subplots(figsize=(11, 6))
ts_sorted = sorted(event_coefs.keys())
coef_vals = [event_coefs[t] for t in ts_sorted]
ci_low = [event_cis[t][0] for t in ts_sorted]
ci_high = [event_cis[t][1] for t in ts_sorted]

pre_mask = [t < 0 for t in ts_sorted]
post_mask = [t >= 0 for t in ts_sorted]
ts_arr = np.array(ts_sorted)

ax.scatter(ts_arr[pre_mask], np.array(coef_vals)[pre_mask],
           color=C_MAIN, s=80, zorder=5, label='政策前 (平行趋势检验)')
ax.scatter(ts_arr[post_mask], np.array(coef_vals)[post_mask],
           color=C_RED, s=80, zorder=5, label='政策后 (动态效应)')

# 画误差条
for i, t in enumerate(ts_sorted):
    color = C_MAIN if t < 0 else C_RED
    ax.plot([t, t], [ci_low[i], ci_high[i]], color=color, linewidth=2)
    ax.plot([t-0.1, t+0.1], [ci_low[i], ci_low[i]], color=color, linewidth=1.5)
    ax.plot([t-0.1, t+0.1], [ci_high[i], ci_high[i]], color=color, linewidth=1.5)

# 基准线
ax.axhline(y=0, color='gray', linewidth=1, linestyle='--', alpha=0.6)
ax.axvline(x=-0.5, color='gray', linewidth=1.5, linestyle=':', alpha=0.5)
ax.text(-0.3, ax.get_ylim()[1]*0.95, '政策实施', fontsize=9, color='gray', ha='left')

ax.set_xlabel('相对于政策实施的时间', fontsize=12)
ax.set_ylabel('系数估计 (βₖ)', fontsize=12)
ax.set_title('图 1：事件研究图——平行趋势与动态效应', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.2)

ax.annotate('基准期\n(t = -1)', xy=(-1, 0), xytext=(-1.5, -0.5),
            fontsize=9, color=C_GRAY,
            arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.2))

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/020-event-study.png', dpi=200, bbox_inches='tight')
print(f"\n✅ 图 1 已保存: 020-event-study.png")

# ========== 3. 平行趋势正式检验 ==========
print("\n平行趋势联合 F 检验...")
pre_vars = [f'treat_x_time_{t}' for t in periods if t < 0 and t != -1]
if pre_vars:
    hypothesis = ' = 0, '.join(pre_vars) + ' = 0'
    try:
        f_test = m_event.f_test(hypothesis)
        print(f"  F 统计量: {f_test.statistic:.3f}")
        print(f"  p 值:     {f_test.pvalue:.4f}")
        if f_test.pvalue > 0.05:
            print("  ✅ 结论: 不能拒绝平行趋势原假设 (p > 0.05)")
        else:
            print("  ⚠️ 结论: 拒绝平行趋势原假设 (p < 0.05)")
    except Exception:
        print("  ⚠️ F 检验无法执行（可能需要调整假设表达）")

# ========== 4. 安慰剂检验 ==========
print("\n执行安慰剂检验 (1000 次随机分配处理时间)...")
np.random.seed(42)

# 先跑真实 DID
X_did = sm.add_constant(df[['treat', 'post']])
X_did['treat_x_post'] = X_did['treat'] * X_did['post']
m_did = sm.OLS(df['y'], X_did).fit()
true_estimate = m_did.params['treat_x_post']
print(f"  真实 DID 估计值: {true_estimate:.4f}")

placebo_estimates = []
for rep in range(1000):
    df_placebo = df.copy()
    fake_treat_time = np.random.choice(periods, size=N)
    df_placebo['fake_post'] = 0
    for u in units:
        mask = df_placebo['unit'] == u
        tt = fake_treat_time[u]
        df_placebo.loc[mask, 'fake_post'] = (df_placebo.loc[mask, 'time'] >= tt).astype(int)

    X_placebo = sm.add_constant(df_placebo[['treat', 'fake_post']])
    X_placebo['treat_x_fake_post'] = X_placebo['treat'] * X_placebo['fake_post']
    m_placebo = sm.OLS(df_placebo['y'], X_placebo).fit()
    placebo_estimates.append(m_placebo.params['treat_x_fake_post'])

placebo_arr = np.array(placebo_estimates)
p_placebo = np.mean(placebo_arr >= true_estimate)
print(f"  安慰剂估计均值: {placebo_arr.mean():.4f}")
print(f"  安慰剂估计标准差: {placebo_arr.std():.4f}")
print(f"  安慰剂 p 值 (真实 >= 随机): {p_placebo:.4f}")
if p_placebo < 0.05:
    print("  ✅ 结论: 真实效应显著大于随机分配产生的效应 (p < 0.05)")
else:
    print("  ⚠️ 结论: 真实效应不显著异于随机分配")

# ===== 图 2：安慰剂分布图 =====
fig, ax = plt.subplots(figsize=(11, 6))
ax.hist(placebo_arr, bins=40, color=C_MAIN, alpha=0.7, edgecolor='white',
        linewidth=0.5, label='安慰剂 DID 估计 (1000 次)')
ax.axvline(x=true_estimate, color=C_RED, linewidth=2.5, linestyle='-',
           label=f'真实 DID 估计 = {true_estimate:.2f}')
ax.axvline(x=placebo_arr.mean(), color=C_GRAY, linewidth=1.5, linestyle='--',
           alpha=0.7, label=f'安慰剂均值 = {placebo_arr.mean():.2f}')
ax.text(0.95, 0.95, f'安慰剂 p 值 = {p_placebo:.3f}', transform=ax.transAxes,
        fontsize=11, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('DID 估计值', fontsize=12)
ax.set_ylabel('频数', fontsize=12)
ax.set_title('图 2：安慰剂检验——随机分配处理时间后的 DID 估计值分布',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/020-placebo-distribution.png', dpi=200, bbox_inches='tight')
print(f"\n✅ 图 2 已保存: 020-placebo-distribution.png")

# ========== 5. 平行趋势原始趋势图 ==========
# ===== 图 3：预处理趋势对比 =====
grouped = df.groupby(['time', 'treat'])['y'].mean().reset_index()

fig, ax = plt.subplots(figsize=(11, 6))
for treat_val, label, color, marker in [
    (0, '对照组 (control)', C_MAIN, 'o'),
    (1, '处理组 (treated)', C_ACCENT, 's')]:
    subset = grouped[grouped['treat'] == treat_val]
    ax.plot(subset['time'], subset['y'], marker, color=color,
            label=label, linewidth=2.5, markersize=8, markerfacecolor=color)

# 政策前趋势拟合（展示平行）
pre_grouped = grouped[grouped['time'] < 0]
for treat_val, color in [(0, C_MAIN), (1, C_ACCENT)]:
    sub = pre_grouped[pre_grouped['treat'] == treat_val]
    if len(sub) > 2:
        z = np.polyfit(sub['time'], sub['y'], 1)
        p_line = np.poly1d(z)
        x_fit = np.linspace(-5, 0, 20)
        ax.plot(x_fit, p_line(x_fit), color=color, linewidth=1.5,
                linestyle='--', alpha=0.5)

ax.axvline(x=0, color='gray', linewidth=2, linestyle='--', alpha=0.7)
ax.text(0.1, ax.get_ylim()[1]*0.95, '政策实施', fontsize=10, color='gray', ha='left')

# 标注平行趋势检验窗口
ax.axvspan(-5, 0, alpha=0.05, color=C_GREEN, label='平行趋势检验窗口')

ax.set_xlabel('时间（相对于政策）', fontsize=12)
ax.set_ylabel('结果变量 Y', fontsize=12)
ax.set_title('图 3：平行趋势检验——政策前两组趋势对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/020-parallel-check.png', dpi=200, bbox_inches='tight')
print(f"\n✅ 图 3 已保存: 020-parallel-check.png")

# ========== 6. 汇总 ==========
print(f"\n{'='*50}")
print("第 20 篇所有图表已生成完毕！")
print(f"  1. 事件研究图:    {IMAGE_DIR}/020-event-study.png")
print(f"  2. 安慰剂分布图:  {IMAGE_DIR}/020-placebo-distribution.png")
print(f"  3. 平行趋势图:    {IMAGE_DIR}/020-parallel-check.png")
print(f"{'='*50}")
