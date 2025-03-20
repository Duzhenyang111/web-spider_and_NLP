import requests
import csv
f = open('评论数据data.csv', mode='w', encoding='utf-8-sig', newline='')
csv_writer = csv.DictWriter(f, fieldnames=[
    '昵称',
    '性别',
    'IP',
    '评论',
    '点赞',
])
csv_writer.writeheader()
headers = {
    "cookie":"i-wanna-go-back=-1; buvid4=5A0F9E46-B442-5656-6F06-F77994AA81C816224-022071221-AYi90B/Us1aMuW8UG5UURA%3D%3D; CURRENT_FNVAL=4048; DedeUserID=38102769; DedeUserID__ckMd5=48197f02d4cf7cc3; home_feed_column=4; buvid3=5104586B-2288-75AD-0019-DC8AAD9657D291971infoc; b_nut=1700312691; b_ut=5; _uuid=76E4C39A-8989-7EBC-AA103-2E429CEB5C2594894infoc; buvid_fp=656c6aee97f239b6c0fd2b2c20de96a7; enable_web_push=DISABLE; rpdid=|(u))|Yk~ul|0J'u~ukRYY~uu; header_theme_version=CLOSE; browser_resolution=1232-598; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MTk1NTY1NjIsImlhdCI6MTcxOTI5NzMwMiwicGx0IjotMX0.IMNHvswFclD4TOpGS6aQYC6XN_XAIX1UwCE0vve6GHA; bili_ticket_expires=1719556502; SESSDATA=a08ce5d8%2C1734849371%2Ca51ef%2A62CjAcICiMCqqdrlLXHlsofQREq_CL3aoyeczHStV4q_ze41TwibuApM5F5KwWe0Srti8SVnFMVXdKTG9sM0VoQ3pad243MGIzWnREa0c5WkM0SE5jdjNSaEdzTG9JRUFJQ0t5TW5uX0FjemRGSzN0cEtHU0V1ZlJqV1FrSERDaW9VRE1UZl9keGJnIIEC; bili_jct=8b7e8ed46cf69ce43c3c193dacee204c; sid=89e3827i; bsource=search_bing; bp_t_offset_38102769=947365323303026688; b_lsid=B2964EBA_190551740FC",
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0",
}
url = 'https://api.bilibili.com/x/v2/reply/wbi/main?oid=613923442&type=1&mode=3&pagination_str=%7B%22offset%22:%22%22%7D&plat=1&seek_rpid=&web_location=1315875&w_rid=94af479db1fba9e0d8b6d32286f403a5&wts=1719416423'
response=requests.get(url=url,headers=headers)
json_data = response.json()
replies = json_data['data']['replies']
print(json_data['data']['replies'])
for index in replies:
    message=index['content']['message']
    like =index['like']
    name = index['member']['uname']
    location=index['reply_control']['location']
    sex = index['member']['sex']

import numpy as np
import re,jieba
from itertools import chain
from snownlp import SnowNLP # 此处使用snownlp模块对评论进行情感分析
import wordcloud
import pandas as pd
import jieba
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image

# 读取包含中文文本的数据
data = pd.read_excel("D:/研究生课程/工作内容/新建文件夹/工作内容.xlsx", header=0)
data = data.applymap(lambda x: x.upper() if isinstance(x, str) else x)
# 停用词列表，可以根据需求扩展
stoplist= [word.strip() for word in open('stopwords.txt',encoding='utf-8').readlines()]

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

from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
# 合并所有分词结果为一个列表
all_tokens = [token for sublist in data['tokenized_text'] for token in sublist]

# 统计词频
word_freq = Counter(all_tokens)

wordcloud = WordCloud(font_path="猴尊宋体.ttf",background_color="white",width=1800,height=1800, margin=5).generate_from_frequencies(word_freq)
#wordcloud = WordCloud(font_path="猴尊宋体.ttf",background_color="white",mask=mask_array,width=1800,height=1800, margin=5).generate_from_frequencies(word_freq)
# 显示词云图
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("wordcloud.png")


def remove(text):
    text1 = re.sub(r'[^\w\s]','', text)
    text2 = re.sub(r'\d+','', text1)
    data = ''.join([item for item in text2])
    text3 = re.findall('[\u4e00-\u9fa5]+', data, re.S)
    text4 = "".join(text3)
    return text
df = data
data['评论'] = data['评论'].apply(remove)
data['score_pl']=data['评论'].apply(lambda x: SnowNLP(x).sentiments)


bins = [0, 0.7, 0.9, 1.0]
labels = ['0-0.7', '0.7-0.9', '0.9-1.0']
data['binned'] = pd.cut(data['score_pl'], bins=bins, labels=labels, include_lowest=True)

# 计算每个区间的频数
bin_counts = data['binned'].value_counts().reindex(labels)

# 绘制柱状图
fig, ax = plt.subplots()

# 设置背景颜色为白色，字体颜色为黑色
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.yaxis.label.set_color('black')
ax.xaxis.label.set_color('black')
ax.title.set_color('black')

ax.bar(bin_counts.index, bin_counts.values, color='black', alpha=0.7)
ax.set_xlabel('Score Range')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of Score PL')
plt.savefig('bar_chart.jpg', format='jpg')
plt.show()
