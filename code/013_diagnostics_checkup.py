"""
知乎计量经济学系列 — 第 13 篇配套代码
主题：诊断总动员——一次完整的回归体检
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings("ignore")

# ===== 统一风格 =====
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
C_MAIN = "#2E86AB"; C_ACCENT = "#F18F01"; C_RED = "#C73E1D"
C_GREEN = "#4C9F70"; C_PURPLE = "#A23B72"; C_GRAY = "#A0A0A0"
IMAGE_DIR = r"C:\Users\qumin\projects\zhihu-econometrics\images"

print("=" * 55)
print("  第 13 篇：诊断总动员——一次完整的回归体检")
print("=" * 55)

# ========== 1. 模拟含多种问题的数据集 ==========
np.random.seed(42)
n = 200

experience = np.random.uniform(1, 40, n)
education = np.random.uniform(8, 22, n)
work_hours = 20 + 2.5 * education + np.random.normal(0, 1.5, n)
age = 22 + experience + np.random.normal(0, 2, n)

errors_raw = np.random.normal(0, 1, n)
errors = np.zeros(n)
errors[0] = errors_raw[0]
for i in range(1, n):
    errors[i] = 0.6 * errors[i-1] + errors_raw[i]
errors = errors * (1 + 0.05 * experience)

salary = (10 + 0.8 * experience + 0.5 * education
          + 0.3 * work_hours + 0.1 * age + errors)

df = pd.DataFrame({
    "experience": experience,
    "education": education,
    "work_hours": work_hours,
    "age": age,
    "salary": salary,
})
print(f"\n📊 数据集: {n} 个观测")
print(df.describe().round(2))

# ========== 2. 图 1：诊断流程图 ==========
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

steps = [
    (1, "① 跑 OLS 回归", "拟合基准模型", C_MAIN),
    (2, "② 残差 vs 拟合值图", "检查异方差（喇叭形）", C_ACCENT),
    (3, "③ VIF 诊断", "检查多重共线性", C_RED),
    (4, "④ Breusch-Pagan 检验", "确认异方差", C_PURPLE),
    (5, "⑤ Durbin-Watson 检验", "检查自相关", C_GREEN),
    (6, "⑥ 综合修复", "稳健标准误 + 变量选择", C_MAIN),
    (7, "⑦ 最终模型", "比较修复前后结果", C_GREEN),
]

for i, (num, title, desc, color) in enumerate(steps):
    y_pos = 9 - i * 1.15
    # Box
    from matplotlib.patches import FancyBboxPatch
    rect = FancyBboxPatch((0.8, y_pos - 0.35), 5.5, 0.7,
                          facecolor=color, alpha=0.15,
                          edgecolor=color, linewidth=2,
                          boxstyle='round,pad=0.05')
    ax.add_patch(rect)
    ax.text(1.0, y_pos, title, fontsize=13, fontweight="bold", va="center")
    ax.text(6.8, y_pos, desc, fontsize=10, color=C_GRAY, va="center")
    if i < len(steps) - 1:
        ax.annotate("", xy=(3.5, y_pos - 0.35), xytext=(3.5, y_pos - 0.35 - 0.8),
                    arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.5))

ax.set_title("回归诊断流程图", fontsize=16, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/013-diagnostic-flowchart.png", dpi=200, bbox_inches="tight")
print("✅ 图 1 已保存：诊断流程图")

# ========== 3. 图 2：2×2 残差诊断面板 ==========
X = sm.add_constant(df[["experience", "education", "work_hours", "age"]])
m1 = sm.OLS(df["salary"], X).fit()
df["resid"] = m1.resid
df["fitted"] = m1.fittedvalues

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) 残差 vs 拟合值
ax = axes[0, 0]
ax.scatter(df["fitted"], df["resid"], alpha=0.5, color=C_MAIN, s=25)
ax.axhline(y=0, color="gray", ls="--", lw=1)
ax.set_xlabel("拟合值"); ax.set_ylabel("残差")
ax.set_title("(a) 残差 vs 拟合值", fontweight="bold"); ax.grid(alpha=0.2)

# (b) 残差 vs experience
ax = axes[0, 1]
ax.scatter(df["experience"], df["resid"], alpha=0.5, color=C_ACCENT, s=25)
ax.axhline(y=0, color="gray", ls="--", lw=1)
ax.set_xlabel("工作经验"); ax.set_ylabel("残差")
ax.set_title("(b) 残差 vs 经验", fontweight="bold"); ax.grid(alpha=0.2)

# (c) 残差的 Q-Q 图
ax = axes[1, 0]
sm.qqplot(df["resid"], line="s", ax=ax, markerfacecolor=C_MAIN,
          markeredgecolor=C_MAIN, alpha=0.6)
ax.set_title("(c) Q-Q 图（正态性）", fontweight="bold"); ax.grid(alpha=0.2)

# (d) 残差序列图（检查自相关）
ax = axes[1, 1]
ax.plot(df.index, df["resid"], color=C_PURPLE, alpha=0.7, lw=0.8)
ax.axhline(y=0, color="gray", ls="--", lw=1)
ax.set_xlabel("观测序号"); ax.set_ylabel("残差")
ax.set_title("(d) 残差序列（自相关）", fontweight="bold"); ax.grid(alpha=0.2)

plt.suptitle("图 2：残差诊断面板", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/013-residual-panel.png", dpi=200, bbox_inches="tight")
print("✅ 图 2 已保存：残差诊断面板")

# ========== 4. VIF 诊断 ==========
print("\n" + "=" * 50)
print("多重共线性诊断 (VIF)")
print("=" * 50)
X_vars = df[["experience", "education", "work_hours", "age"]]
X_vif = sm.add_constant(X_vars)
vif_data = pd.DataFrame()
vif_data["Variable"] = X_vars.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i+1)
                   for i in range(len(X_vars.columns))]
print(vif_data.round(2))
if vif_data["VIF"].max() > 5:
    print("⚠️ 存在严重共线性 (VIF > 5)")
else:
    print("✅ 无严重共线性")

# ========== 5. BP 检验 ==========
print("\n" + "=" * 50)
print("异方差诊断 (Breusch-Pagan 检验)")
print("=" * 50)
bp_test = het_breuschpagan(m1.resid, X)
print(f"  LM 统计量: {bp_test[0]:.2f}")
print(f"  p 值:      {bp_test[1]:.6f}")
if bp_test[1] < 0.05:
    print("  ⚠️ 存在异方差 (p < 0.05)")
else:
    print("  ✅ 同方差假设成立")

# ========== 6. DW 检验 ==========
print("\n" + "=" * 50)
print("自相关诊断 (Durbin-Watson 检验)")
print("=" * 50)
dw = durbin_watson(m1.resid)
print(f"  DW 统计量: {dw:.4f}")
if dw < 1.5:
    print("  ⚠️ 存在正自相关 (DW < 1.5)")
elif dw > 2.5:
    print("  ⚠️ 存在负自相关 (DW > 2.5)")
else:
    print("  ✅ 无显著自相关")

# ========== 7. 图 3：诊断报告汇总 ==========
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")
diagnostics = [
    ("诊断项目", "结果", "结论"),
    ("---" * 8, "---" * 12, "---" * 6),
    ("多重共线性 (VIF)", f"最大 VIF = {vif_data['VIF'].max():.1f}",
     "⚠️ 需处理" if vif_data["VIF"].max() > 5 else "✅ 正常"),
    ("异方差 (BP 检验)", f"LM = {bp_test[0]:.2f}, p = {bp_test[1]:.4f}",
     "⚠️ 存在" if bp_test[1] < 0.05 else "✅ 正常"),
    ("自相关 (DW 检验)", f"DW = {dw:.4f}",
     "⚠️ 存在" if (dw < 1.5 or dw > 2.5) else "✅ 正常"),
    ("模型拟合", f"R² = {m1.rsquared:.3f}, Adj-R² = {m1.rsquared_adj:.3f}",
     "---"),
]
table = ax.table(cellText=diagnostics[1:],
                 colLabels=diagnostics[0],
                 cellLoc="center", loc="center",
                 colWidths=[0.25, 0.35, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor(C_MAIN)
        cell.set_text_props(color="white", fontweight="bold")
    elif "⚠️" in str(cell.get_text().get_text()):
        cell.set_facecolor("#FFF3E0")
    else:
        cell.set_facecolor("#F5F5F5")
ax.set_title("图 3：回归诊断报告 · 汇总", fontsize=15, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/013-diagnostic-report.png", dpi=200, bbox_inches="tight")
print("✅ 图 3 已保存：诊断报告汇总")

# ========== 8. 修复 ==========
print("\n" + "=" * 50)
print("开始修复...")
print("=" * 50)

# 修复 1：删除共线性变量（work_hours 与 education 高度相关）
df_clean = df.drop(columns=["work_hours"])
print("\n📌 修复 1：删除 work_hours（VIF 过高）")

# 修复 2：用稳健标准误
X_clean = sm.add_constant(df_clean[["experience", "education", "age"]])
m_clean = sm.OLS(df_clean["salary"], X_clean).fit()
m_robust = m_clean.get_robustcov_results(cov_type="HC3")

print("📌 修复 2：使用稳健标准误 (HC3)")
print("\n修复后模型摘要：")
print(m_robust.summary())

# 修复后诊断
bp_clean = het_breuschpagan(m_clean.resid, X_clean)
dw_clean = durbin_watson(m_clean.resid)
print(f"\n修复后 BP 检验: LM={bp_clean[0]:.2f}, p={bp_clean[1]:.4f}")
print(f"修复后 DW 检验: {dw_clean:.4f}")

# ========== 9. 图 4：修复前后对比 ==========
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# 左：修复前残差
ax = axes[0]
ax.scatter(df["fitted"], df["resid"], alpha=0.5, color=C_RED, s=25)
ax.axhline(y=0, color="gray", ls="--", lw=1)
ax.set_xlabel("拟合值"); ax.set_ylabel("残差")
ax.set_title(f"修复前\nR²={m1.rsquared:.3f}, BP p={bp_test[1]:.4f}",
             fontweight="bold"); ax.grid(alpha=0.2)

# 右：修复后残差
ax = axes[1]
df_clean["fitted2"] = m_clean.fittedvalues
df_clean["resid2"] = m_clean.resid
ax.scatter(df_clean["fitted2"], df_clean["resid2"], alpha=0.5, color=C_GREEN, s=25)
ax.axhline(y=0, color="gray", ls="--", lw=1)
ax.set_xlabel("拟合值"); ax.set_ylabel("残差")
ax.set_title(f"修复后\nR²={m_clean.rsquared:.3f}, BP p={bp_clean[1]:.4f}",
             fontweight="bold"); ax.grid(alpha=0.2)

plt.suptitle("图 4：修复前后对比", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/013-before-after.png", dpi=200, bbox_inches="tight")
print("✅ 图 4 已保存：修复前后对比")

print("\n" + "=" * 55)
print("  ✅ 第 13 篇所有图表已生成完毕！")
print("=" * 55)
