"""
知乎计量经济学系列 — 第 14 篇配套代码
主题：数据有了时间标签——自相关与 ACF
"""
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

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

# ========== 0. 生成模拟数据 ==========
np.random.seed(42)
n_days = 365 * 3  # 3 年
t = np.arange(n_days)

# 气温：正弦波（年周期）+ 随机噪声
temperature = (
    15 + 15 * np.sin(2 * np.pi * t / 365 - np.pi / 2)
    + np.random.normal(0, 3, n_days)
)

# 白噪声
white_noise = np.random.normal(0, 3, n_days)

# 股票价格（随机游走）
daily_returns = np.random.normal(0.0005, 0.02, n_days)
price = 100 * np.exp(np.cumsum(daily_returns))
log_returns = np.diff(np.log(price))

print(f"✅ 模拟数据已生成：{n_days} 个观测值（约 {n_days/365:.0f} 年）")
print(f"   气温范围: {temperature.min():.1f}°C ~ {temperature.max():.1f}°C")
print(f"   最终股价: {price[-1]:.2f}（起始 100）\n")

# ========== 1. 图 1：气温时间序列 ==========
fig, ax = plt.subplots(figsize=(14, 5))

# 只画出前 2 年使细节更清晰
plot_days = 365 * 2
ax.plot(t[:plot_days], temperature[:plot_days],
        color=C_MAIN, linewidth=0.8, alpha=0.85)

# 标注夏季/冬季
ax.axvspan(0, 90, alpha=0.05, color=C_RED, label='冬季（1-3月）')
ax.axvspan(90, 180, alpha=0.05, color=C_GREEN, label='春季（4-6月）')
ax.axvspan(180, 270, alpha=0.05, color=C_ACCENT, label='夏季（7-9月）')
ax.axvspan(270, 365, alpha=0.05, color=C_PURPLE, label='秋季（10-12月）')
ax.axvspan(365, 455, alpha=0.05, color=C_RED)
ax.axvspan(455, 545, alpha=0.05, color=C_GREEN)
ax.axvspan(545, 635, alpha=0.05, color=C_ACCENT)
ax.axvspan(635, 730, alpha=0.05, color=C_PURPLE)

# 在峰值处添加标注
peak_idx = np.argmax(temperature[:365])
ax.annotate(f'夏季最高 {temperature[peak_idx]:.0f}°C',
            xy=(peak_idx, temperature[peak_idx]),
            xytext=(peak_idx + 30, temperature[peak_idx] + 5),
            fontsize=9, color=C_RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2))

trough_idx = np.argmin(temperature[:365])
ax.annotate(f'冬季最低 {temperature[trough_idx]:.0f}°C',
            xy=(trough_idx, temperature[trough_idx]),
            xytext=(trough_idx + 30, temperature[trough_idx] - 8),
            fontsize=9, color=C_MAIN, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_MAIN, lw=1.2))

ax.set_xlabel('天数（第 1 天 = 1 月 1 日）', fontsize=12)
ax.set_ylabel('气温（°C）', fontsize=12)
ax.set_title('图 1：模拟北京日均气温（前 2 年）—— 清晰的年周期', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='upper right', ncol=2)
ax.grid(alpha=0.2)
ax.set_xlim(0, plot_days)
ax.set_ylim(-15, 35)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/014-temperature-timeseries.png',
            dpi=200, bbox_inches='tight')
print("✅ 图 1 已保存：014-temperature-timeseries.png")

# ========== 2. 图 2：ACF 图 ==========
fig, ax = plt.subplots(figsize=(10, 5))
plot_acf(temperature, lags=40, zero=False, ax=ax,
         color=C_MAIN, markerfacecolor=C_MAIN, markersize=4)

ax.set_xlabel('滞后阶数（天）', fontsize=12)
ax.set_ylabel('自相关系数', fontsize=12)
ax.set_title('图 2：气温的 ACF（Autocorrelation Function）', fontsize=13, fontweight='bold')

# 手动添加滞后标记说明
ax.text(0.95, 0.95, f'lag 1: {acf(temperature, nlags=40)[1]:.3f}',
        transform=ax.transAxes, fontsize=9, color=C_RED, fontweight='bold',
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C_GRAY, alpha=0.8))

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/014-acf-example.png',
            dpi=200, bbox_inches='tight')
print("✅ 图 2 已保存：014-acf-example.png")

# ========== 3. 图 3：白噪声 vs 自相关序列 ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 上左：气温时序
ax1 = axes[0, 0]
ax1.plot(t[:500], temperature[:500], color=C_MAIN, linewidth=0.8)
ax1.set_title('（a）气温序列（有自相关）', fontsize=11, fontweight='bold')
ax1.set_ylabel('°C', fontsize=10)
ax1.grid(alpha=0.2)

# 上右：气温 ACF
ax2 = axes[0, 1]
plot_acf(temperature, lags=30, zero=False, ax=ax2,
         color=C_MAIN, markerfacecolor=C_MAIN, markersize=3)
ax2.set_title('（b）气温 ACF', fontsize=11, fontweight='bold')
ax2.set_ylim(-0.3, 1.1)

# 下左：白噪声时序
ax3 = axes[1, 0]
ax3.plot(t[:500], white_noise[:500], color=C_GRAY, linewidth=0.6)
ax3.set_title('（c）白噪声序列（无自相关）', fontsize=11, fontweight='bold')
ax3.set_ylabel('值', fontsize=10)
ax3.set_xlabel('天数', fontsize=10)
ax3.grid(alpha=0.2)

# 下右：白噪声 ACF
ax4 = axes[1, 1]
plot_acf(white_noise, lags=30, zero=False, ax=ax4,
         color=C_GRAY, markerfacecolor=C_GRAY, markersize=3)
ax4.set_title('（d）白噪声 ACF', fontsize=11, fontweight='bold')
ax4.set_ylim(-0.3, 1.1)
ax4.set_xlabel('滞后阶数', fontsize=10)

plt.suptitle('图 3：自相关序列 vs 白噪声——一个有"记忆"，一个纯随机',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/014-white-noise.png',
            dpi=200, bbox_inches='tight')
print("✅ 图 3 已保存：014-white-noise.png")

# 输出白噪声的 ACF 值验证
wn_acf = acf(white_noise, nlags=5)
print(f"   白噪声 ACF(lag 1-5): {[f'{v:.3f}' for v in wn_acf[1:6]]}")

# ========== 4. 图 4：股票价格 ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# 上左：价格时序
ax1 = axes[0, 0]
ax1.plot(t, price, color=C_RED, linewidth=0.7, alpha=0.85)
ax1.set_title('（a）模拟股票价格（随机游走）', fontsize=11, fontweight='bold')
ax1.set_ylabel('价格（元）', fontsize=10)
ax1.grid(alpha=0.2)

# 上右：价格 ACF
ax2 = axes[0, 1]
plot_acf(price, lags=40, zero=False, ax=ax2,
         color=C_RED, markerfacecolor=C_RED, markersize=3)
ax2.set_title('（b）价格的 ACF——非平稳导致的缓慢衰减', fontsize=11, fontweight='bold')
ax2.set_ylim(-0.3, 1.1)

# 下左：收益率时序
ax3 = axes[1, 0]
ax3.plot(t[1:], log_returns, color=C_GREEN, linewidth=0.5, alpha=0.7)
ax3.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
ax3.set_title('（c）对数收益率（≈白噪声）', fontsize=11, fontweight='bold')
ax3.set_ylabel('收益率', fontsize=10)
ax3.set_xlabel('天数', fontsize=10)
ax3.grid(alpha=0.2)

# 下右：收益率 ACF
ax4 = axes[1, 1]
plot_acf(log_returns, lags=40, zero=False, ax=ax4,
         color=C_GREEN, markerfacecolor=C_GREEN, markersize=3)
ax4.set_title('（d）收益率 ACF——绝大多数在蓝带内', fontsize=11, fontweight='bold')
ax4.set_xlabel('滞后阶数', fontsize=10)
ax4.set_ylim(-0.3, 1.1)

plt.suptitle('图 4：股票价格的"特殊气质"——价格非平稳，收益率≈白噪声',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/014-stock-price.png',
            dpi=200, bbox_inches='tight')
print("✅ 图 4 已保存：014-stock-price.png")

# 输出收益率 ACF 对比
ret_acf = acf(log_returns, nlags=3)
print(f"   收益率 ACF(lag 1-3): {[f'{v:.4f}' for v in ret_acf[1:4]]}")

# ===== 额外输出：ACF 数值表 =====
temp_acf = acf(temperature, nlags=10)
print(f"\n📊 气温 ACF 数值（lag 0-10）：")
print(f"   {'lag':>4} {'ACF':>8}")
print(f"   {'---':>4} {'---':>8}")
for k in range(11):
    print(f"   {k:>4} {temp_acf[k]:>8.4f}")

print(f"\n🎉 第 14 篇所有图表已生成完毕！")
