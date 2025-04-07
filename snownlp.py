import re
import jieba
import collections
import openpyxl
import stylecloud
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from snownlp import SnowNLP # 此处使用snownlp模块对评论进行情感分析
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from itertools import chain




# 读取包含中文文本的数据
data = pd.read_excel("D:/市调/工作内容.xlsx", header=0)
data = data.applymap(lambda x: x.upper() if isinstance(x, str) else x)
# 停用词列表，可以根据需求扩展
stoplist= [word.strip() for word in open('stopwords.txt',encoding='utf-8').readlines()]

custom_words = [
    '海底椰',
    'linlee',
    '手打柠檬茶',
    '大绿鸭',
    '椰汁青柠',
    '盒马',
    '霸王茶姬',
    '鸭屎香',
    '冰柠',
    '好喝',
    '一点点',
    '小黄鸭',
    '柠季',
    '曾舜晞'
    # 添加更多的自定义词组...
]

# 将自定义词组添加到jieba的词典中
for word in custom_words:
    jieba.add_word(word)

# 定义分词函数，并过滤停用词
def tokenize_text(text):
    # 使用jieba进行分词
    tokens = jieba.lcut(text)
    # 过滤掉空格和停用词
    tokens = [token.strip() for token in tokens if token.strip() and token.strip() not in stoplist]
    return tokens

# 对数据中的文本列应用分词函数
data['tokenized_text'] = data['评论'].apply(tokenize_text)


df = data
def remove(text):
    text1 = re.sub(r'[^\w\s]','', text)
    text2 = re.sub(r'\d+','', text1)
    data = ''.join([item for item in text2])
    text3 = re.findall('[\u4e00-\u9fa5]+', data, re.S)
    text4 = "".join(text3)
    return text4
data['评论'] = data['评论'].apply(remove)
data['score_pl']=data['评论'].apply(lambda x: SnowNLP(x).sentiments)
#data = data[data['score_pl'] != 1.0]
data.to_excel('output.xlsx', index=False) # 保存修改后的数据到新的Excel文件

# 定义分词函数
def tokenize(text):
    return " ".join(jieba.cut(text))
object_list = df['评论'].apply(tokenize)
df['评论分词'] = df['评论'].apply(tokenize)
negPath = r"负面情绪词.txt"
posPath = r"正面情绪词.txt"
pos = open(posPath, encoding='utf-8').readlines()
neg = open(negPath, encoding='utf-8').readlines()
#统计积极、消极词
for i in range(len(pos)):
    pos[i] = pos[i].replace('\n','').replace('\ufeff','')
for i in range(len(neg)):
    neg[i] = neg[i].replace('\n','').replace('\ufeff','')

# 假设 df 是包含评论内容的 DataFrame，评论内容所在列名为 '分词后的评论内容'
# 将评论内容分词，并转换为列表
tokenized_comments = df['评论分词'].apply(lambda x: x.split())
 
# 初始化正面词和负面词的词频字典
pos_word_freq = {}
neg_word_freq = {}
 
# 遍历分词结果列表，计算正面词和负面词的词频
for tokens in tokenized_comments:
    for token in tokens:
        if token in pos:
            pos_word_freq[token] = pos_word_freq.get(token, 0) + 1
        elif token in neg:
            neg_word_freq[token] = neg_word_freq.get(token, 0) + 1
 
# 将词频字典转换为 pandas Series
pos_word_freq_series = pd.Series(pos_word_freq).sort_values(ascending=False)
neg_word_freq_series = pd.Series(neg_word_freq).sort_values(ascending=False)
 
# 显示正面词和负面词的词频分布
print("正面词词频分布：")
print(pos_word_freq_series)
print("\n负面词词频分布：")
print(neg_word_freq_series)
pos_word_freq_series.to_excel("积极.xlsx")
neg_word_freq_series.to_excel("消极.xlsx")

tokenized_comments = df['评论分词'].apply(lambda x: x.split())

# 初始化正面词和负面词的词频字典
# 定义分箱区间
bins = [0, 0.7, 0.9, 1.0]
labels = ['0-0.7', '0.7-0.9', '0.9-1.0']

# 将'score_pl'分箱
data['binned'] = pd.cut(data['score_pl'], bins=bins, labels=labels, include_lowest=True)

# 计算每个区间的频数
bin_counts = data['binned'].value_counts().reindex(labels)

# 绘制柱状图
plt.bar(bin_counts.index, bin_counts.values, color='blue', alpha=0.7)
plt.xlabel('Score Range')
plt.ylabel('Frequency')
plt.title('Histogram of Score PL')
plt.show()
