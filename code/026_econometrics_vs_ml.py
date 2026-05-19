"""
知乎计量经济学系列 — 第 26 篇配套代码
主题：计量 vs 机器学习——哪个更准？
对比：OLS (statsmodels) vs Ridge vs Lasso vs XGBoost (或 RandomForest)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
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
COLORS = [C_MAIN, C_ACCENT, C_GREEN, C_PURPLE]
IMAGE_DIR = r'C:\Users\qumin\projects\zhihu-econometrics\images'
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ========== 1. 模拟房价数据（线性 + 非线性交互结构） ==========
n = 1000

# --- 线性特征 ---
area = np.random.normal(120, 30, n)          # 面积（m²）
bedrooms = np.random.randint(1, 6, n)         # 卧室数
floor = np.random.randint(1, 35, n)           # 楼层
location = np.random.uniform(0, 10, n)        # 地段评分 0-10
age = np.random.uniform(0, 30, n)             # 房龄（年）

# --- 非线性交互特征 ---
# 面积 × 地段交互：好地段的大房子有额外溢价
interaction_1 = (area / 100) * location * 12
# 卧室 × 楼层交互：高层的大户型更受欢迎
interaction_2 = bedrooms * (floor / 10) * 5
# 非线性特征：房龄的衰减效应（新房折旧快，老房折旧慢）
age_nonlinear = -np.log(age + 1) * 15

# --- 噪声变量（ML 需要抵抗的干扰） ---
noise_1 = np.random.normal(0, 1, n)
noise_2 = np.random.normal(0, 1, n)
noise_3 = np.random.normal(0, 1, n)
noise_4 = np.random.normal(0, 1, n)

# --- 真实房价生成 ---
price = (
    30                           # 基准价
    + 3.0 * area                  # 面积效应（线性）
    + 10 * bedrooms               # 卧室效应（线性）
    + 0.4 * floor                 # 楼层效应（线性）
    + 18 * location               # 地段效应（线性）
    - 2.5 * age                   # 房龄效应（线性）
    + interaction_1               # 交互项1
    + interaction_2               # 交互项2
    + age_nonlinear               # 非线性项
    + np.random.normal(0, 25, n)  # 随机误差
)

df = pd.DataFrame({
    'area': area, 'bedrooms': bedrooms, 'floor': floor,
    'location': location, 'age': age,
    'noise_1': noise_1, 'noise_2': noise_2,
    'noise_3': noise_3, 'noise_4': noise_4,
    'price': price
})

FEATURES = ['area', 'bedrooms', 'floor', 'location', 'age',
            'noise_1', 'noise_2', 'noise_3', 'noise_4']
TARGET = 'price'

print(f"✅ 模拟数据生成：{n} 条，{len(FEATURES)} 个特征（含线性+交互+非线性+噪声）")
print(f"   房价均值: {price.mean():.1f} 万元, 标准差: {price.std():.1f} 万元")
print(f"   真实关系包含 2 个非线性交互项 + 1 个对数衰减项\n")

# ========== 2. 训练/测试分割 ==========
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED
)
print(f"训练集: {len(X_train)} 条 | 测试集: {len(X_test)} 条\n")

# ========== 3. 模型训练 ==========

# --- 3a. OLS (statsmodels) ---
print("=" * 55)
print("模型训练中...")
print("=" * 55)

X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

ols_model = sm.OLS(y_train, X_train_sm).fit()
y_pred_ols = ols_model.predict(X_test_sm)
rmse_ols = sqrt(mean_squared_error(y_test, y_pred_ols))
print(f"  ✅ OLS 完成 | RMSE = {rmse_ols:.2f}")

# --- 3b. Ridge ---
ridge_model = Ridge(alpha=5.0, random_state=RANDOM_SEED)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
rmse_ridge = sqrt(mean_squared_error(y_test, y_pred_ridge))
print(f"  ✅ Ridge 完成 | RMSE = {rmse_ridge:.2f} | alpha=5.0")

# --- 3c. Lasso ---
lasso_model = Lasso(alpha=2.0, random_state=RANDOM_SEED, max_iter=10000)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
rmse_lasso = sqrt(mean_squared_error(y_test, y_pred_lasso))
print(f"  ✅ Lasso 完成 | RMSE = {rmse_lasso:.2f} | alpha=2.0")

# --- 3d. RandomForest（替代 XGBoost） ---
rf_model = RandomForestRegressor(
    n_estimators=200, max_depth=10,
    min_samples_leaf=5, random_state=RANDOM_SEED
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rmse_rf = sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"  ✅ RandomForest 完成 | RMSE = {rmse_rf:.2f} | n_estimators=200")

print()

# ========== 4. 图 1：模型 RMSE 对比条形图 ==========
models = ['OLS', 'Ridge', 'Lasso', 'RandomForest']
rmses = [rmse_ols, rmse_ridge, rmse_lasso, rmse_rf]
bar_colors = [C_MAIN, C_ACCENT, C_GREEN, C_PURPLE]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, rmses, color=bar_colors, alpha=0.85,
              edgecolor='white', linewidth=1.5, width=0.6)

# 在柱子顶部标数值
for bar, val in zip(bars, rmses):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold',
            color='#333333')

# 标注 OLS 和 ML 的差异
ax.annotate('计量学派\n可解释性强', xy=(0, rmse_ols),
            xytext=(-0.3, rmse_ols + 18),
            fontsize=9, color=C_MAIN, fontweight='bold',
            ha='center',
            arrowprops=dict(arrowstyle='->', color=C_MAIN, lw=1.5))
ax.annotate('机器学习派\n预测精度高', xy=(3, rmse_rf),
            xytext=(3.3, rmse_rf - 15),
            fontsize=9, color=C_PURPLE, fontweight='bold',
            ha='center',
            arrowprops=dict(arrowstyle='->', color=C_PURPLE, lw=1.5))

ax.set_ylabel('测试集 RMSE（万元）', fontsize=12)
ax.set_title('图 1：各模型在测试集上的预测精度对比', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.2)
ax.set_ylim(0, max(rmses) * 1.35)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/026-model-comparison.png', dpi=200, bbox_inches='tight')
print("✅ 图 1 已保存：模型 RMSE 对比条形图")

# ========== 5. 图 2：实际 vs 预测散点图 ==========
predictions = {
    'OLS': y_pred_ols,
    'Ridge': y_pred_ridge,
    'Lasso': y_pred_lasso,
    'RandomForest': y_pred_rf
}

fig, axes = plt.subplots(2, 2, figsize=(12, 11))
axes = axes.flatten()

for idx, (name, y_pred) in enumerate(predictions.items()):
    ax = axes[idx]
    ax.scatter(y_test, y_pred, alpha=0.4, s=15, color=COLORS[idx],
               edgecolors='white', linewidth=0.3)

    # 完美预测参考线 y=x
    lim_min = min(y_test.min(), y_pred.min()) - 10
    lim_max = max(y_test.max(), y_pred.max()) + 10
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

    rmse_val = sqrt(mean_squared_error(y_test, y_pred))
    ax.text(0.05, 0.9, f'RMSE = {rmse_val:.1f}', transform=ax.transAxes,
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel('实际房价（万元）', fontsize=10)
    ax.set_ylabel('预测房价（万元）', fontsize=10)
    ax.set_title(f'({chr(97+idx)}) {name}', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.15)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

plt.suptitle('图 2：各模型预测值 vs 实际值散点图', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/026-prediction-scatter.png', dpi=200, bbox_inches='tight')
print("✅ 图 2 已保存：预测-实际散点图")

# ========== 6. 图 3：可解释性对比 ==========
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# --- (a) OLS 系数（带 95% CI） ---
ax1 = axes[0]
ols_coefs = ols_model.params.drop('const')
ols_ci = ols_model.conf_int().drop('const', axis=0)
ols_ci_lower = ols_ci[0]
ols_ci_upper = ols_ci[1]

# 按系数绝对值排序
coef_order = np.argsort(np.abs(ols_coefs.values))
coef_names_sorted = [FEATURES[i] for i in coef_order]
coef_vals_sorted = ols_coefs.values[coef_order]
ci_low_sorted = ols_ci_lower.values[coef_order]
ci_high_sorted = ols_ci_upper.values[coef_order]

y_pos = np.arange(len(coef_names_sorted))
ax1.errorbar(coef_vals_sorted, y_pos,
             xerr=[coef_vals_sorted - ci_low_sorted, ci_high_sorted - coef_vals_sorted],
             fmt='o', color=C_MAIN, capsize=3, capthick=1.5, markersize=8)
ax1.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(coef_names_sorted, fontsize=9)
ax1.set_xlabel('系数估计值', fontsize=11)
ax1.set_title('(a) OLS 系数（带 95% CI）', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.2)

# --- (b) Ridge 系数 ---
ax2 = axes[1]
ridge_coefs = ridge_model.coef_
ridge_order = np.argsort(np.abs(ridge_coefs))
ax2.barh(range(len(FEATURES)), ridge_coefs[ridge_order],
         color=C_ACCENT, alpha=0.8, edgecolor='white')
ax2.set_yticks(range(len(FEATURES)))
ax2.set_yticklabels([FEATURES[i] for i in ridge_order], fontsize=9)
ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel('Ridge 系数', fontsize=11)
ax2.set_title('(b) Ridge 系数（L2 收缩）', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.2)

# --- (c) RandomForest 特征重要性 ---
ax3 = axes[2]
rf_importances = rf_model.feature_importances_
rf_order = np.argsort(rf_importances)
ax3.barh(range(len(FEATURES)), rf_importances[rf_order],
         color=C_PURPLE, alpha=0.8, edgecolor='white')
ax3.set_yticks(range(len(FEATURES)))
ax3.set_yticklabels([FEATURES[i] for i in rf_order], fontsize=9)
ax3.set_xlabel('特征重要性', fontsize=11)
ax3.set_title('(c) RandomForest 特征重要性', fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.2)

plt.suptitle('图 3：可解释性对比——OLS 系数 vs Ridge 系数 vs 特征重要性', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/026-interpretability.png', dpi=200, bbox_inches='tight')
print("✅ 图 3 已保存：可解释性对比图")

# ========== 7. 输出详细结果对比表 ==========
print("\n" + "=" * 55)
print("结果对比汇总")
print("=" * 55)
print(f"{'模型':<15} {'测试集 RMSE':<15} {'vs OLS 改善':<15}")
print("-" * 45)
print(f"{'OLS':<15} {rmse_ols:<15.2f} {'-':<15}")
print(f"{'Ridge':<15} {rmse_ridge:<15.2f} {((rmse_ridge-rmse_ols)/rmse_ols)*100:<+14.1f}%")
print(f"{'Lasso':<15} {rmse_lasso:<15.2f} {((rmse_lasso-rmse_ols)/rmse_ols)*100:<+14.1f}%")
print(f"{'RandomForest':<15} {rmse_rf:<15.2f} {((rmse_rf-rmse_ols)/rmse_ols)*100:<+14.1f}%")
print("-" * 45)

# 输出 OLS 系数 p 值（看看哪些变量显著）
print("\nOLS 回归结果（部分）：")
ols_summary = pd.DataFrame({
    '系数': ols_coefs,
    'p值': ols_model.pvalues.drop('const'),
    '显著': ['***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else '' for p in ols_model.pvalues.drop('const')]
})
print(ols_summary.to_string())

# 输出 Ridge 和 RF 的非零系数
print("\nRidge 系数（绝对值 > 0.5）：")
ridge_s = pd.Series(ridge_model.coef_, index=FEATURES)
print(ridge_s[ridge_s.abs() > 0.5].to_string())

print("\nRandomForest 特征重要性（> 1%）：")
rf_s = pd.Series(rf_model.feature_importances_, index=FEATURES)
print(rf_s[rf_s > 0.01].to_string())

print(f"\n🎉 第 26 篇所有图表已生成完毕！")
print(f"   关键结论：OLS 虽精度稍低，但系数有明确的统计推断；")
print(f"   RF 精度最高，但内部机制如同黑箱。")