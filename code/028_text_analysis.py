"""
知乎计量经济学系列 — 第 28 篇配套代码
主题：把文字变成数字——文本分析的计量入门
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import jieba
import statsmodels.api as sm
from collections import Counter

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

# ===== 固定随机种子 =====
np.random.seed(42)

# ========== 1. 构建情感词典 ==========
print("=" * 60)
print("第 28 篇：文本分析的计量入门")
print("=" * 60)

positive_words = [
    '好吃', '满意', '推荐', '赞', '不错', '实惠', '新鲜', '快', '好',
    '喜欢', '值得', '划算', '方便', '良心', '惊喜', '惊艳', '细腻',
    '丰富', '干净', '热情', '舒服', '正宗', '超值', '好评', '回购',
    '品质', '信赖', '放心', '优秀', '出色', '精致', '完美', '很棒'
]

negative_words = [
    '差', '一般', '失望', '慢', '贵', '难吃', '差评', '不推荐',
    '敷衍', '变质', '少', '脏', '态度差', '后悔', '不值', '垃圾',
    '恶心', '糟糕', '凑合', '无语', '坑', '骗人', '问题', '投诉',
    '不新鲜', '退款', '批评', '恶劣', '极差', '投诉', '差劲'
]

# 构建情感得分字典
sentiment_dict = {}
for w in positive_words:
    sentiment_dict[w] = 1
for w in negative_words:
    sentiment_dict[w] = -1

print(f"情感词典：{len(positive_words)} 个正面词 + {len(negative_words)} 个负面词")

# ========== 2. 模拟中文商品评论 ==========
review_templates = [
    # (正面模板, 负面模板, 中性模板)
    ("这家店的东西真的{pos1}，{pos2}！", "这家店的东西太{neg1}了，{neg2}。", "还行吧，一般般。"),
    ("{pos1}，价格也{pos2}，推荐购买。", "价格{neg1}，东西也{neg2}，不推荐。", "价格适中，凑合用。"),
    ("第二次买了，{pos1}一如既往地{pos2}。", "不会再买了，一次比一次{neg1}。", "买过几次，没什么变化。"),
    ("包装{pos1}，物流{pos2}，好评！", "包装{neg1}，物流{neg2}，差评！", "包装和物流都一般。"),
    ("{pos1}出乎意料地{pos2}，下次还来。", "{neg1}，完全不{neg2}，后悔了。", "没什么特别的感觉。"),
    ("质量{pos1}，客服态度也{pos2}，满分。", "质量{neg1}，客服态度{neg2}，投诉。", "质量一般，客服没联系。"),
    ("味道{pos1}，分量{pos2}，性价比超高！", "味道{neg1}，分量{neg2}，不值这个价。", "味道还行，分量一般。"),
    ("已经推荐给朋友了，大家都说{pos1}。", "朋友推荐来的，但很{neg1}，尴尬。", "朋友推荐的，一般般。"),
    ("没想到这么{pos1}，完全超出预期！", "没想到这么{neg1}，和描述完全不符。", "和预期差不多，不惊喜。"),
    ("{pos1}，{pos2}，总之非常满意。", "{neg1}，{neg2}，总之非常失望。", "还行，不算差也不算好。"),
]

def generate_review(sentiment_score):
    """根据情感得分生成一条中文评论"""
    template_group = review_templates[np.random.randint(0, len(review_templates))]

    if sentiment_score > 0.3:
        pos1 = np.random.choice(positive_words[:15], 2, replace=False)
        review = template_group[0].format(pos1=pos1[0], pos2=pos1[1])
    elif sentiment_score < -0.3:
        neg1 = np.random.choice(negative_words[:15], 2, replace=False)
        review = template_group[1].format(neg1=neg1[0], neg2=neg1[1])
    else:
        review = template_group[2]

    # 随机追加额外情感词
    if np.random.random() < 0.4:
        extra = np.random.choice(positive_words + negative_words, 1)[0]
        review += f" {extra}。"

    return review

# 生成 200 条评论
n_samples = 200
true_sentiments = np.random.uniform(-2, 2, n_samples)

reviews = []
for i in range(n_samples):
    s = true_sentiments[i]
    # 加点噪声让评论不完全对应情感分数
    noisy_s = s + np.random.normal(0, 0.5)
    reviews.append(generate_review(noisy_s))

# ========== 3. jieba 分词 + 情感打分 ==========
print("\n正在用 jieba 进行分词和情感分析……")

def compute_sentiment(text):
    """对一条评论做分词并计算情感得分"""
    words = jieba.lcut(text)
    score = 0
    word_count = 0
    for w in words:
        if w in sentiment_dict:
            score += sentiment_dict[w]
            word_count += 1
    # 如果有情感词，取平均；否则为 0
    return score / word_count if word_count > 0 else 0

sentiment_scores = []
for r in reviews:
    sentiment_scores.append(compute_sentiment(r))

sentiment_scores = np.array(sentiment_scores)

# ========== 4. 模拟销量数据 ==========
# 销量 = 基线 + 情感得分效应 + 价格效应 + 促销效应 + 噪声
base_sales = 500
price = np.random.uniform(20, 200, n_samples)
promotion = np.random.binomial(1, 0.3, n_samples)  # 是否促销

sales = (
    base_sales
    + 80 * sentiment_scores
    - 1.5 * price
    + 120 * promotion
    + np.random.normal(0, 80, n_samples)
)
sales = np.maximum(sales, 0)  # 销量不能为负

# 构造 DataFrame
df = pd.DataFrame({
    'review': reviews,
    'sentiment_score': sentiment_scores,
    'price': price,
    'promotion': promotion,
    'sales': sales
})

print(f"共处理 {len(df)} 条评论")
print(f"情感得分范围: [{df['sentiment_score'].min():.2f}, {df['sentiment_score'].max():.2f}]")
print(f"销量范围: [{df['sales'].min():.0f}, {df['sales'].max():.0f}]")

# ========== 5. 图 1：情感词典词频统计（条形图，不是词云） ==========
print("\n绘制图 1：情感词典词频统计……")

# 统计所有评论中情感词的出现频率
word_counter = Counter()
for r in reviews:
    words = jieba.lcut(r)
    for w in words:
        if w in sentiment_dict:
            word_counter[w] += 1

# 分别取正面和负面出现最多的前 10 个词
pos_counts = {w: word_counter[w] for w in positive_words if w in word_counter}
neg_counts = {w: word_counter[w] for w in negative_words if w in word_counter}

top_pos = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)[:10]
top_neg = sorted(neg_counts.items(), key=lambda x: x[1], reverse=True)[:10]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 正面词
if top_pos:
    words_p, counts_p = zip(*top_pos)
    axes[0].barh(range(len(words_p)), counts_p, color=C_GREEN, alpha=0.8, edgecolor='white')
    axes[0].set_yticks(range(len(words_p)))
    axes[0].set_yticklabels(words_p, fontsize=11)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('出现次数', fontsize=11)
    axes[0].set_title('正面情感词 TOP 10', fontsize=13, fontweight='bold', color=C_GREEN)
    axes[0].grid(axis='x', alpha=0.3)

# 负面词
if top_neg:
    words_n, counts_n = zip(*top_neg)
    axes[1].barh(range(len(words_n)), counts_n, color=C_RED, alpha=0.8, edgecolor='white')
    axes[1].set_yticks(range(len(words_n)))
    axes[1].set_yticklabels(words_n, fontsize=11)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('出现次数', fontsize=11)
    axes[1].set_title('负面情感词 TOP 10', fontsize=13, fontweight='bold', color=C_RED)
    axes[1].grid(axis='x', alpha=0.3)

plt.suptitle('图 1：评论中情感词出现频率', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/028-wordcloud.png', dpi=200, bbox_inches='tight')
print("✅ 图 1 已保存：情感词频统计条形图")

# ========== 6. 图 2：情感得分 vs 销量（散点图 + 回归线） ==========
print("\n绘制图 2：情感得分 vs 销量……")

X_simple = sm.add_constant(df['sentiment_score'])
m_simple = sm.OLS(df['sales'], X_simple).fit()

fig, ax = plt.subplots(figsize=(10, 6))

# 散点
ax.scatter(df['sentiment_score'], df['sales'], alpha=0.5,
           color=C_MAIN, edgecolors='white', linewidth=0.3, s=40)

# 回归线
xp = np.linspace(df['sentiment_score'].min() - 0.2, df['sentiment_score'].max() + 0.2, 200)
yp = m_simple.params.iloc[0] + m_simple.params.iloc[1] * xp
ax.plot(xp, yp, color=C_RED, linewidth=2.5,
        label=f'销量 = {m_simple.params.iloc[0]:.1f} + {m_simple.params.iloc[1]:.1f} × 情感得分')

# 置信带
from statsmodels.sandbox.regression.predstd import wls_prediction_std
_, upper, lower = wls_prediction_std(m_simple, exog=sm.add_constant(xp))
ax.fill_between(xp, lower, upper, color=C_RED, alpha=0.08, label='95% 置信带')

ax.set_xlabel('情感得分（nlp 计算）', fontsize=12)
ax.set_ylabel('月销量', fontsize=12)
ax.set_title(f'图 2：情感得分 vs 销量 (R² = {m_simple.rsquared:.3f})',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/028-sentiment-vs-sales.png', dpi=200, bbox_inches='tight')
print("✅ 图 2 已保存：情感得分 vs 销量散点图")

# ========== 7. 完整回归：情感 + 价格 + 促销 ==========
print("\n" + "=" * 60)
print("回归分析：销量 ~ 情感得分 + 价格 + 促销")
print("=" * 60)

X_multi = sm.add_constant(df[['sentiment_score', 'price', 'promotion']])
m_multi = sm.OLS(df['sales'], X_multi).fit()
print(m_multi.summary())

# ========== 8. 图 3：回归结果可视化（森林图风格） ==========
print("\n绘制图 3：回归结果可视化……")

coef_df = pd.DataFrame({
    'Variable': ['截距', '情感得分', '价格', '促销'],
    'Coefficient': m_multi.params.values,
    'SE': m_multi.bse.values,
    'P>|t|': m_multi.pvalues.values
})
coef_df['CI_Lower'] = coef_df['Coefficient'] - 1.96 * coef_df['SE']
coef_df['CI_Upper'] = coef_df['Coefficient'] + 1.96 * coef_df['SE']
coef_df['Significant'] = coef_df['P>|t|'] < 0.05

fig, ax = plt.subplots(figsize=(10, 6))

# 只显示非截距系数
plot_df = coef_df.iloc[1:]

y_pos = range(len(plot_df))

# Plot each point separately to handle significance colors for error bars
for i, (_, row) in enumerate(plot_df.iterrows()):
    color = C_MAIN if row['Significant'] else C_GRAY
    ax.errorbar(row['Coefficient'], i,
                xerr=1.96 * row['SE'],
                fmt='o', color=color, ecolor=color, capsize=5,
                markersize=10, linewidth=2)

# 添加参考线（零线）
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

# 标注数值
for i, (_, row) in enumerate(plot_df.iterrows()):
    label = f"{row['Coefficient']:.1f} (p={row['P>|t|']:.4f})"
    ax.text(row['Coefficient'] + 1.5 * row['SE'] + 3, i,
            label, va='center', fontsize=10,
            color=C_MAIN if row['Significant'] else C_GRAY)

ax.set_yticks(list(y_pos))
ax.set_yticklabels(plot_df['Variable'], fontsize=12)
ax.set_xlabel('系数估计值 (95% CI)', fontsize=12)
ax.set_title('图 3：回归系数图——各变量对销量的影响', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.2)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/028-regression-results.png', dpi=200, bbox_inches='tight')
print("✅ 图 3 已保存：回归系数森林图")

# ========== 9. 打印几条示例评论 ==========
print("\n" + "=" * 60)
print("示例评论 + 情感分析结果")
print("=" * 60)
np.random.seed(None)
example_indices = np.random.choice(n_samples, 5, replace=False)
for idx in example_indices:
    row = df.iloc[idx]
    print(f"\n📝 评论: {row['review']}")
    print(f"   → 分词后情感得分: {row['sentiment_score']:+.2f}")
    print(f"   → 实际销量: {row['sales']:.0f}")

# ========== 10. 保存数据供参考 ==========
df.to_csv(f'{IMAGE_DIR}/../code/review_sentiment_data.csv', index=False, encoding='utf-8-sig')
print(f"\n💾 数据已保存至 code/review_sentiment_data.csv")
print("🎉 第 28 篇所有图表已生成完毕！")
