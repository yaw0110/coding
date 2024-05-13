#分析和解释数据：利用excel对原始时间格式进行调整，利用pyecharts画图，查看政策文件发布热度
from pyecharts import options as opts
from pyecharts.charts import Bar
import pandas as pd
from pyecharts.commons.utils import JsCode
from pyecharts.faker import Faker

data=pd.read_excel(r'G:\网络爬虫比赛\url1.xlsx','url')['时间调整']
data_count=data.value_counts()

c = (
    Bar()
    .add_xaxis(data_count.index.values.tolist())
    .add_yaxis("各月（从多到少）政策文件数", data_count.values.tolist())
    .set_global_opts(
        title_opts=opts.TitleOpts(title="", subtitle=""),
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
    .render(r'G:\网络爬虫比赛\各月政策文件数.html'))