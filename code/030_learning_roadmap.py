"""
知乎计量经济学系列 — 第 30 篇配套代码
主题：从零到计量——你的学习地图
功能：生成七大能力图谱（学习路线图/技能树可视化）
"""
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适用于无 GUI 环境

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ===== 统一风格 =====
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

C_MAIN = '#2E86AB'
C_ACCENT = '#F18F01'
C_RED = '#C73E1D'
C_GREEN = '#4C9F70'
C_PURPLE = '#A23B72'
C_TEAL = '#1B998B'
C_GRAY = '#A0A0A0'

IMAGE_DIR = r'C:\Users\qumin\projects\zhihu-econometrics\images'

# ===== 定义七大能力模块 =====
abilities = [
    {
        'name': '回归分析',
        'name_en': 'Regression',
        'articles': '第 1-9, 27 篇',
        'skills': 'OLS · 多元回归 · 虚拟变量\n交互效应 · Logit 模型',
        'color': C_MAIN,
        'pos': (0.0, 0.0),
    },
    {
        'name': '诊断检验',
        'name_en': 'Diagnostics',
        'articles': '第 10-13 篇',
        'skills': '异方差 · 多重共线性\nVIF · AIC/BIC · 残差分析',
        'color': C_ACCENT,
        'pos': (1.5, 0.8),
    },
    {
        'name': '时间序列',
        'name_en': 'Time Series',
        'articles': '第 14-18 篇',
        'skills': '平稳性 · ADF 检验\nARIMA · GARCH · 分解',
        'color': C_RED,
        'pos': (3.0, 0.0),
    },
    {
        'name': '因果推断',
        'name_en': 'Causal Inference',
        'articles': '第 19-22 篇',
        'skills': 'DID · 平行趋势 · RDD\n工具变量 IV · 2SLS',
        'color': C_GREEN,
        'pos': (4.5, 0.8),
    },
    {
        'name': '面板数据',
        'name_en': 'Panel Data',
        'articles': '第 23-24 篇',
        'skills': '固定效应 FE · 随机效应 RE\nHausman 检验',
        'color': C_PURPLE,
        'pos': (6.0, 0.0),
    },
    {
        'name': '编程实现',
        'name_en': 'Coding',
        'articles': '贯穿全系列',
        'skills': 'Python · statsmodels\nlinearmodels · 可视化',
        'color': C_TEAL,
        'pos': (1.5, -1.8),
    },
    {
        'name': '结果解读',
        'name_en': 'Interpretation',
        'articles': '贯穿全系列',
        'skills': '系数解读 · 置信区间\n统计显著 vs 经济显著',
        'color': '#D4A5A5',
        'pos': (4.5, -1.8),
    },
]

# ===== 连接关系 =====
# (from_index, to_index)
connections = [
    (0, 1),   # 回归 → 诊断
    (0, 2),   # 回归 → 时间序列
    (1, 3),   # 诊断 → 因果推断
    (2, 4),   # 时间序列 → 面板数据
    (3, 4),   # 因果推断 → 面板数据
    (0, 5),   # 回归 → 编程实现
    (1, 5),   # 诊断 → 编程实现
    (3, 6),   # 因果推断 → 结果解读
    (4, 6),   # 面板数据 → 结果解读
]

# ===== 绘制 =====
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(-1.5, 8.0)
ax.set_ylim(-3.0, 2.5)
ax.set_aspect('equal')
ax.axis('off')

# ----- 绘制连接线（先画线，再画节点，确保节点在顶层）-----
for from_i, to_i in connections:
    p1 = np.array(abilities[from_i]['pos'])
    p2 = np.array(abilities[to_i]['pos'])

    # 计算控制点（向上弯曲）
    mid = (p1 + p2) / 2
    # 根据连接类型选择弯曲方向
    if abs(p1[0] - p2[0]) > 1.5:  # 横向连接
        ctrl = mid + np.array([0, 0.5])
    else:
        ctrl = mid + np.array([0.3, 0.3])

    # 使用贝塞尔曲线
    t_vals = np.linspace(0, 1, 100)
    curve = np.zeros((100, 2))
    for i, t in enumerate(t_vals):
        curve[i] = (1 - t)**2 * p1 + 2 * (1 - t) * t * ctrl + t**2 * p2

    ax.plot(curve[:, 0], curve[:, 1],
            color=C_GRAY, linewidth=1.5, alpha=0.4,
            linestyle='--', zorder=1)

# ----- 绘制节点 -----
for ability in abilities:
    x, y = ability['pos']
    color = ability['color']

    # 外圈发光效果（绘制多个同心圆）
    for r in [0.75, 0.72]:
        circle = plt.Circle((x, y), r, color=color,
                            alpha=0.08 if r > 0.72 else 0.15,
                            zorder=2)
        ax.add_patch(circle)

    # 主圆形
    circle = plt.Circle((x, y), 0.65, color=color,
                        alpha=0.85, ec='white', linewidth=2.5,
                        zorder=3)
    ax.add_patch(circle)

    # 能力名称（中文）
    ax.text(x, y + 0.1, ability['name'],
            ha='center', va='center',
            fontsize=13, fontweight='bold',
            color='white', zorder=4)

    # 能力名称（英文小字）
    ax.text(x, y - 0.25, ability['name_en'],
            ha='center', va='center',
            fontsize=7.5, fontstyle='italic',
            color='white', alpha=0.8, zorder=4)

# ----- 技能详情文本框 -----
for ability in abilities:
    x, y = ability['pos']
    color = ability['color']

    # 文本框位置：在节点下方
    box_y = y - 1.1

    # 带圆角的文本框
    bbox_props = dict(
        boxstyle='round,pad=0.5',
        facecolor='white',
        edgecolor=color,
        linewidth=1.5,
        alpha=0.9,
    )

    text_content = (
        f"{ability['articles']}\n"
        f"{ability['skills']}"
    )

    ax.text(x, box_y, text_content,
            ha='center', va='top',
            fontsize=8.5, color='#333333',
            linespacing=1.5,
            bbox=bbox_props, zorder=5)

# ----- 阶段标注（顶部大括号式标注）-----
phase_labels = [
    ('直觉建立', 0.0, 2.0, C_MAIN),
    ('核心工具箱', 0.8, 1.8, C_ACCENT),
    ('时间序列', 2.3, 1.5, C_RED),
    ('因果推断', 3.8, 1.8, C_GREEN),
    ('实战专题', 5.5, 2.0, C_PURPLE),
]

for label, x_start, width, color in phase_labels:
    rect = mpatches.FancyBboxPatch(
        (x_start - 0.3, 1.6), width, 0.5,
        boxstyle="round,pad=0.1",
        facecolor=color, alpha=0.15,
        edgecolor=color, linewidth=1.0,
        zorder=1,
    )
    ax.add_patch(rect)
    ax.text(x_start + width / 2 - 0.3, 1.85, label,
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            color=color, alpha=0.8, zorder=2)

# ----- 标题 -----
ax.text(3.0, 2.35, '📊 计量经济学七大能力图谱',
        ha='center', va='center',
        fontsize=20, fontweight='bold',
        color='#2C3E50', zorder=10)

ax.text(3.0, 2.08, '从零到计量——你的技能树全景图',
        ha='center', va='center',
        fontsize=11, color=C_GRAY,
        fontstyle='italic', zorder=10)

# ----- 页脚说明 -----
ax.text(3.0, -2.8,
        '五大阶段：直觉建立 → 核心工具箱 → 时间序列 → 因果推断 → 实战专题     共 30 篇文章',
        ha='center', va='bottom',
        fontsize=9, color=C_GRAY, alpha=0.7)

ax.text(3.0, -3.0,
        '© 像普林斯顿微积分一样学计量经济学 · 系列终章',
        ha='center', va='bottom',
        fontsize=8, color=C_GRAY, alpha=0.5)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/030-roadmap.png',
            dpi=250, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"✅ 学习路线图已保存至: {IMAGE_DIR}/030-roadmap.png")

# ===== 额外：生成一张交互式 HTML 学习路径图（文字版） =====
print("\n" + "=" * 65)
print("你的学习路径摘要")
print("=" * 65)
print()
print("📌 七大能力模块及对应文章：")
print()

for i, ability in enumerate(abilities, 1):
    print(f"  {i}. {ability['name']} ({ability['name_en']})")
    print(f"     涉及文章: {ability['articles']}")
    print(f"     核心技能: {ability['skills'].replace(chr(10), ' · ')}")
    print()

print("📌 推荐学习顺序：")
print()
print("  第一阶段: 回归分析 + 结果解读（同时进行）")
print("  第二阶段: 诊断检验")
print("  第三阶段: 时间序列")
print("  第四阶段: 因果推断 → 面板数据")
print("  第五阶段: 编程实现（贯穿始终）")
print()
print("📌 推荐资源：")
print()
print("  教材: Wooldridge → Angrist → Cunningham")
print("  课程: Coursera (Erasmus) + Bilibili (陈强)")
print("  工具: Python (statsmodels) → R (fixest) → Stata")
print()
print("🎉 第 30 篇所有内容已就绪！")
