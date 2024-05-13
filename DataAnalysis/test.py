import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties

# 假设 'WenQuanYi Zen Hei' 是你要使用的中文字体
# font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
# font_prop = FontProperties(fname=font_path, size=14)

# 设置 matplotlib 使用中文字体
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 替换为实际的中文字体名称
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# matplotlib.rcParams['font.sans-serif'] = ['wqy-microhei.ttc']  # 替换为您系统中已有的字体
# matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# plt.title('中文标题', fontproperties=font_prop)
# plt.xlabel('X轴标签', fontproperties=font_prop)
# plt.ylabel('Y轴标签', fontproperties=font_prop)

plt.title('中文标题')
plt.xlabel('X轴标签')
plt.ylabel('Y轴标签')

# 显示图形
plt.show()