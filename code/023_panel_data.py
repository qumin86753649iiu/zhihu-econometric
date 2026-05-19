"""
知乎计量经济学系列 — 第 23 篇配套代码
主题：面板数据——同一批人，看很久
方法：Pooled OLS, Between Estimator, Within (FE) Estimator
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

# ===== 统一风格 =====
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
C_MAIN = '#2E86AB'
C_ACCENT = '#F18F01'
C_RED = '#C73E1D'
C_GREEN = '#4C9F70'
C_PURPLE = '#A23B72'
C_GRAY = '#A0A0A0'
C_LIGHT_BLUE = '#D4E8F0'
IMAGE_DIR = r'C:\Users\qumin\projects\zhihu-econometrics\images'

np.random.seed(42)
N, T = 100, 5

print("=" * 55)
print("第 23 篇：面板数据——同一批人，看很久")
print("=" * 55)

# ========== 1. 模拟面板数据 ==========
print("\n>>> 正在模拟面板数据...")
print(f"   个体数 N = {N}, 时间 T = {T}, 总观测 = {N * T}")

# 1a. 个体效应（不可观测的"管理水平"）
alpha = np.random.normal(0, 2, N)

# 1b. 时变变量
rd_staff = np.random.randint(5, 100, (N, T))
leverage = np.random.uniform(0.2, 0.7, (N, T))

# 1c. 不随时间变化的变量
industry = np.random.choice(['科技', '制造', '消费'], N)
founder_ceo = np.random.binomial(1, 0.4, N)

# 1d. 生成利润（含个体效应和行业效应）
profit = np.zeros((N, T))
for i in range(N):
    for t in range(T):
        profit[i, t] = (
            5
            + 0.08 * rd_staff[i, t]
            - 3.0 * leverage[i, t]
            + (2.0 if industry[i] == '科技' else -1.0)
            + alpha[i]
            + np.random.normal(0, 1.5)
        )

# 1e. 整理成 Long-Format 面板
df_list = []
for i in range(N):
    for t in range(T):
        df_list.append({
            'firm_id': f'F{i+1:03d}',
            'year': 2019 + t,
            'profit': profit[i, t],
            'rd_staff': float(rd_staff[i, t]),
            'leverage': leverage[i, t],
            'industry': industry[i],
            'founder_ceo': founder_ceo[i],
        })
df = pd.DataFrame(df_list)
print(f"   数据框形状: {df.shape}")
print(f"   字段: {list(df.columns)}")

# ========== 2. 图 1：面板数据结构示意图 ==========
print("\n>>> 绘制面板数据结构示意图...")
fig, ax = plt.subplots(figsize=(10, 7))

# 取前 6 家公司画成一个网格表
sample_firms = df['firm_id'].unique()[:6]
sample_data = df[df['firm_id'].isin(sample_firms)]

# 绘制网格
for idx, fid in enumerate(sample_firms):
    firm_data = sample_data[sample_data['firm_id'] == fid]
    for tidx, (_, row) in enumerate(firm_data.iterrows()):
        x = tidx
        y = idx
        profit_val = row['profit']
        # 颜色深浅表示利润高低
        color_intensity = (profit_val - df['profit'].min()) / (df['profit'].max() - df['profit'].min())
        rect_color = plt.cm.Blues(0.2 + 0.6 * color_intensity)
        ax.add_patch(plt.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                                    facecolor=rect_color, edgecolor='white', linewidth=1.5))
        ax.text(x, y, f'{profit_val:.1f}', ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')

# 标注
ax.set_xlim(-0.6, T - 0.4)
ax.set_ylim(-0.6, len(sample_firms) - 0.4)
ax.set_xticks(range(T))
ax.set_xticklabels([f'year {2019 + t}' for t in range(T)], fontsize=10)
ax.set_yticks(range(len(sample_firms)))
ax.set_yticklabels([f'公司 {fid}' for fid in sample_firms], fontsize=10)
ax.set_xlabel('时间 t', fontsize=12)
ax.set_ylabel('个体 i', fontsize=12)
ax.set_title('图 1：面板数据结构——个体 i × 时间 t', fontsize=14, fontweight='bold')

# 添加变量类型标注
ax.annotate('时变变量\n(profit, rd_staff, leverage)',
            xy=(T + 0.5, 2.5), fontsize=10, color=C_MAIN,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=C_LIGHT_BLUE, alpha=0.8))
ax.annotate('不随时间变化\n(industry, founder_ceo)',
            xy=(T + 0.5, 0.5), fontsize=10, color=C_RED,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5D0C5', alpha=0.8))

# 添加 colorbar 说明
sm_scatter = ax.scatter([], [], c=[], cmap='Blues', vmin=df['profit'].min(), vmax=df['profit'].max())
cbar = plt.colorbar(sm_scatter, ax=ax, fraction=0.02, pad=0.02)
cbar.set_label('利润', fontsize=10)

ax.grid(False)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/023-panel-structure.png', dpi=200, bbox_inches='tight')
print("✅ 图 1 已保存：面板数据结构示意图")

# ========== 3. 三种方法估计 ==========
print("\n>>> 开始估计三种面板数据模型...")

# --- 3a. Pooled OLS ---
X_p = sm.add_constant(df[['rd_staff', 'leverage']])
m_pooled = sm.OLS(df['profit'], X_p).fit(cov_type='HC3')
print("\n--- Pooled OLS 结果 ---")
print(m_pooled.summary().tables[1])

# --- 3b. Between Estimator ---
between = df.groupby('firm_id')[['profit', 'rd_staff', 'leverage']].mean().reset_index()
X_b = sm.add_constant(between[['rd_staff', 'leverage']])
m_between = sm.OLS(between['profit'], X_b).fit(cov_type='HC3')
print("\n--- Between Estimator 结果 ---")
print(m_between.summary().tables[1])

# --- 3c. Within Estimator (FE) ---
df_panel = df.set_index(['firm_id', 'year'])
m_within = PanelOLS.from_formula(
    'profit ~ rd_staff + leverage + EntityEffects',
    data=df_panel
).fit(cov_type='robust')
print("\n--- Within (FE) Estimator 结果 ---")
print(m_within)

# ========== 4. 系数对比表 ==========
results = {
    'Pooled OLS': {'rd_staff': m_pooled.params['rd_staff'],
                   'leverage': m_pooled.params['leverage']},
    'Between':    {'rd_staff': m_between.params['rd_staff'],
                   'leverage': m_between.params['leverage']},
    'Within (FE)': {'rd_staff': m_within.params['rd_staff'],
                    'leverage': m_within.params['leverage']},
}

print("\n" + "=" * 55)
print("系数估计值对比：")
print(f"{'方法':>12} {'研发系数':>10} {'负债率系数':>12}")
print("-" * 36)
for method, coefs in results.items():
    print(f"{method:>12} {coefs['rd_staff']:>10.4f} {coefs['leverage']:>12.3f}")
print(f"{'真实值':>12} {'0.0800':>10} {'-3.000':>12}")
print("=" * 55)

# ========== 5. 图 2：Pooled OLS vs Within 系数对比 =====
print("\n>>> 绘制系数对比图...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

methods = ['Pooled OLS', 'Between', 'Within (FE)']
x = np.arange(len(methods))
width = 0.35

# 左：研发系数
ax = axes[0]
rd_vals = [results[m]['rd_staff'] for m in methods]
bars = ax.bar(x, rd_vals, width, color=[C_MAIN, C_ACCENT, C_GREEN], alpha=0.8, edgecolor='white')
# 真实值线
ax.axhline(y=0.08, color=C_RED, linewidth=2, linestyle='--', label=f'真实值 = 0.080')
# 标注数值
for bar, val in zip(bars, rd_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel('研发人数系数', fontsize=12)
ax.set_title('(a) 研发人数 → 利润', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.2)

# 右：负债率系数
ax = axes[1]
lev_vals = [results[m]['leverage'] for m in methods]
bars = ax.bar(x, lev_vals, width, color=[C_MAIN, C_ACCENT, C_GREEN], alpha=0.8, edgecolor='white')
ax.axhline(y=-3.0, color=C_RED, linewidth=2, linestyle='--', label=f'真实值 = -3.000')
for bar, val in zip(bars, lev_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel('负债率系数', fontsize=12)
ax.set_title('(b) 负债率 → 利润', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.2)

plt.suptitle('图 2：三种方法系数对比——Within (FE) 最接近真实值', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/023-pooled-vs-within.png', dpi=200, bbox_inches='tight')
print("✅ 图 2 已保存：系数对比图")

# ========== 6. 图 3：个体轨迹图 ==========
print("\n>>> 绘制个体轨迹图...")
fig, ax = plt.subplots(figsize=(12, 7))

# 所有公司的轨迹：浅灰色表示
for fid in df['firm_id'].unique():
    firm_data = df[df['firm_id'] == fid]
    ax.plot(firm_data['year'], firm_data['profit'],
            color=C_GRAY, alpha=0.15, linewidth=0.8)

# 挑选 5 家有代表性的公司高亮
highlight_firms = ['F001', 'F015', 'F033', 'F067', 'F092']
highlight_colors = [C_MAIN, C_RED, C_GREEN, C_PURPLE, C_ACCENT]

for fid, color in zip(highlight_firms, highlight_colors):
    firm_data = df[df['firm_id'] == fid]
    ax.plot(firm_data['year'], firm_data['profit'],
            color=color, linewidth=2.5, marker='o', markersize=6,
            label=f'公司 {fid}', zorder=5)
    # 在终点标注公司名
    last_row = firm_data.iloc[-1]
    ax.annotate(fid, (last_row['year'], last_row['profit']),
                xytext=(5, 0), textcoords='offset points',
                fontsize=9, fontweight='bold', color=color)

ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('利润', fontsize=12)
ax.set_title('图 3：100 家公司的利润轨迹——每条线都是一个"个体"的故事', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper left', ncol=2)
ax.grid(alpha=0.15)
ax.set_xticks(range(2019, 2024))

# 添加文字说明
ax.text(0.02, 0.98,
        '浅灰色线 = 100 家公司轨迹\n彩色线 = 5 家代表性公司',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/023-individual-trajectories.png', dpi=200, bbox_inches='tight')
print("✅ 图 3 已保存：个体轨迹图")

# ========== 7. 额外信息：数据概览 ==========
print("\n" + "=" * 55)
print("数据概览：")
print(f"   公司数: {df['firm_id'].nunique()}")
print(f"   年份范围: {df['year'].min()} - {df['year'].max()}")
print(f"   行业分布: {df.groupby('industry')['firm_id'].nunique().to_dict()}")
print(f"   创始人任CEO比例: {df['founder_ceo'].mean():.1%}")
print(f"\n   各变量描述性统计：")
print(df[['profit', 'rd_staff', 'leverage']].describe().round(3))

print("\n" + "=" * 55)
print("🎉 第 23 篇全部图表已生成完毕！")
print("=" * 55)
