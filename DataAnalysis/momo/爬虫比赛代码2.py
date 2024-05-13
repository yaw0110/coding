#利用jieba进行分词，词频统计，筛选出名词，利用pyecharts画出不同词性的的柱状图
#总体进行词频统计
import codecs
import pandas as pd
from collections import Counter
import jieba.posseg as pseg
#读入文件
import os

path = "G:\网络爬虫比赛\momo\data" #文件夹目录

files= os.listdir(path) #得到文件夹下的所有文件名称

txts = []

for file in files: #遍历文件夹

    position = path+'\\'+ file #构造绝对路径，"\\"，其中一个'\'为转义符

    with open(position, "r",encoding='utf-8') as f: #打开文件

        data = f.read() #读取文件

        txts.append(data)

txts = ','.join(txts)#转化为非数组类型
#分词并词性标注
x3=list(pseg.cut(txts))
#统计词频
x4=Counter(x3)
y1=[]
y2=[]
y3=[]

for j in x4:
    y1.append(x4[j])
    y2.append(list(j)[0])
    y3.append(list(j)[1])

z2=pd.DataFrame({'词频':y1,'分词':y2,'词性':y3})
z2=z2.sort_values(by='词频',ascending=False)
z2.to_excel('G:\网络爬虫比赛\政策文件内容词频统计.xlsx',sheet_name='Sheet1',index=False)

import os
from pyecharts import options as opts
from pyecharts.charts import Bar,Page
from pyecharts.charts import Bar
from pyecharts.commons.utils import JsCode
from pyecharts.faker import Faker
pos=['n','ng','nrfg','nrt','ns','nt','nz']
page = Page()
i = "政策文件内容词频"
i1=pd.read_excel('G:\网络爬虫比赛\政策文件内容词频统计.xlsx','Sheet1')
i2=i1[i1['词性'].isin(pos)].iloc[0:20,:]
i3=(
    Bar()
    .add_xaxis(i2['分词'].to_list())
    .add_yaxis(i,i2['词频'].to_list())
    .set_global_opts(
        title_opts=opts.TitleOpts(title=""),
        datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
    )
    .set_series_opts(
        itemstyle_opts={
            "normal": {
                "color": JsCode(
                    """new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                offset: 0,
                color: 'rgba(0, 244, 255, 1)'
            }, {
                offset: 1,
                color: 'rgba(0, 77, 167, 1)'
            }], false)"""
                ),
                "barBorderRadius": [30, 30, 30, 30],
                "shadowColor": "rgb(0, 160, 221)",
            }
        }
    )
)
page.add(i3)
page.render('G:\网络爬虫比赛\政策文件内容词频统计.html')