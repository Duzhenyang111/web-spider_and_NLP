import numpy as np
import re
from itertools import chain
import pandas as pd
import jieba

# 读取包含中文文本的数据
data = pd.read_excel("exmaple.xlsx", header=0)
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



from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
# 合并所有分词结果为一个列表
all_tokens = [token for sublist in data['tokenized_text'] for token in sublist]

# 统计词频
word_freq = Counter(all_tokens)


mask_image = Image.open("enlarged_image.png")
mask_array = np.array(mask_image)
# 生成词云
# wordcloud = WordCloud(font_path="猴尊宋体.ttf",background_color="white",width=1800,height=1800, margin=5).generate_from_frequencies(word_freq)
wordcloud = WordCloud(font_path="猴尊宋体.ttf",background_color="white",mask=mask_array,width=1800,height=1800, margin=5).generate_from_frequencies(word_freq)
# 显示词云图
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("wordcloud.png")