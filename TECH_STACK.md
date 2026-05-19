# 技术栈与发布方案

## 一、代码/图表生成工作流

### 每篇文章的生成流水线
```
Plan阶段 → 写MD正文 → 生成Python图 → 生概念图 → 发布知乎
```

### Python 数据图统一模板
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm

# 统一字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# 统一配色
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#4C9F70', '#C73E1D']

# 统一图形风格
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = '#F8F8F8'
plt.rcParams['axes.facecolor'] = '#F8F8F8'
```

### 概念图生成方案
1. **简单示意图**：用 matplotlib 的 `patches` + `annotate` 手绘（适合因果图、DAG）
2. **流程图/路线图**：用 Mermaid.js 生成 SVG
3. **封面图**：每篇一个定制化概念图，用 DALL-E 3 或 Midjourney 生成（提示词模板固定）

### 封面图生成提示词模板（DALL-E/Midjourney）
```
A clean, minimalist illustration of [concept], educational style, 
warm color palette (blue, orange, white), 
no text, suitable for a Chinese popular science article on Zhihu, 
flat vector art style, professional looking
```

---

## 二、发布流程

### 发布前检查清单
- [ ] Markdown 排版无误（标题层级、代码块标记）
- [ ] Python 代码跑通，输出截图保存
- [ ] 数据图生成完毕，导为 PNG/HiDPI（300 DPI）
- [ ] 概念图/封面图生成完毕
- [ ] 所有图片上传图床（建议 sm.ms / 阿里OSS）
- [ ] 替换文中图片引用为图床链接
- [ ] 知乎富文本粘贴后预览一次（重点检查代码高亮和表格）
- [ ] 日期时间确认：周二/四/日 20:00

### 图床策略
- **优先**：GitHub + jsDelivr CDN（免费、稳定）
- **备选**：sm.ms 图床（国内可访问）
- 图片命名：`econometrics-{序号}-{图序号}.png`

### 每篇文章的 GitHub Repo 结构
```
zhihu-econometrics/
├── README.md               # 项目总介绍
├── OUTLINE.md              # 完整大纲
├── TECH_STACK.md           # 技术方案（本文件）
├── articles/                # 文章源码（Markdown）
│   ├── 001-计量经济学到底在干什么.md
│   ├── 002-那条线是怎么画出来的.md
│   └── ...
├── code/                    # 每篇附的 Python 代码
│   ├── 001_coffee_productivity.py
│   ├── 002_least_squares.py
│   └── ...
├── images/                  # 生成的图表原文件
│   ├── 001-scatter.png
│   ├── 001-regression.png
│   └── ...
└── notebooks/              # Jupyter Notebook 完整版
    └── 001-coffee-productivity.ipynb
```

---

## 三、知乎发布注意事项

### 排版适配
- 知乎支持 Markdown 部分语法，但 **表格和代码块** 需要手动调整
- 代码块建议用知乎的代码高亮功能（3 个反引号 + 语言标识）
- 公式用知乎的 LaTeX（`$...$` 行内，`$$...$$` 块级）
- 图片宽高建议：**800px × 自适应**，不超过知乎正文最大宽度

### SEO 标题策略
- 主标题不超过 30 字，要包含核心关键词
- 副标题增加长尾关键词覆盖
- 示例：
  - "计量经济学到底在干什么？——从一个咖啡问题说起"
  - 覆盖关键词：计量经济学入门、相关与因果、回归分析

### 互动设计
- 每篇结尾留 2 个开放性问题
- 评论区积极回复，第 1 条评论自己留（引导方向）
- 设置 1 个数据/代码挑战（如"用你的数据跑一遍，把结果截图发评论区"）
