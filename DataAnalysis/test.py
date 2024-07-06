# import matplotlib.pyplot as plt
# import matplotlib
# from matplotlib.font_manager import FontProperties

# # 假设 'WenQuanYi Zen Hei' 是你要使用的中文字体
# # font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
# # font_prop = FontProperties(fname=font_path, size=14)

# # 设置 matplotlib 使用中文字体
# matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 替换为实际的中文字体名称
# matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# # matplotlib.rcParams['font.sans-serif'] = ['wqy-microhei.ttc']  # 替换为您系统中已有的字体
# # matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# # plt.title('中文标题', fontproperties=font_prop)
# # plt.xlabel('X轴标签', fontproperties=font_prop)
# # plt.ylabel('Y轴标签', fontproperties=font_prop)

# plt.title('中文标题')
# plt.xlabel('X轴标签')
# plt.ylabel('Y轴标签')

# # 显示图形
# plt.show()
import time

sql1 = '''
    select recordid odsrecordid, tenantcode, submittime, concat(toString(storeid), '-', toString(tenantcode)) storeid, storename, photo,  phototype, oss from ods_store_inandout_photo
    global inner join ods_tenant t on t.code = tenantcode
    where photo != ''
'''

sql2 = '''
    select recordid odsrecordid, tenantcode, signintime submittime, concat(toString(storeid), '-', toString(tenantcode)) storeid, storename, signinpicture photo, 1 phototype, oss from ods_store_inandout
    global inner join ods_tenant t on t.code = tenantcode
    where photo != ''
'''

sql2_1 = f'''
    {sql2} 
    and submittime <= '2023-06-01 00:00:00'
'''

sql2_2 = f'''
    {sql2} 
    and submittime <= '2023-10-01 00:00:00' and submittime > '2023-06-01 00:00:00'
'''

sql2_3 = f'''
    {sql2} 
    and  submittime <= '2024-01-01 00:00:00' and submittime > '2023-10-01 00:00:00'
'''

sql2_4 = f'''
    {sql2} 
    and  submittime <= '2024-06-01 00:00:00' and submittime > '2024-01-01 00:00:00'
'''

sql1_1 = f'''
    {sql1} and submittime <= '2023-06-01 00:00:00'
'''

sql1_2 = f'''
    {sql1} 
    and submittime <= '2023-10-01 00:00:00' and submittime > '2023-06-01 00:00:00'
'''

sql1_3 = f'''
    {sql1} 
    and  submittime <= '2024-01-01 00:00:00' and submittime > '2023-10-01 00:00:00'
'''

sql1_4 = f'''
    {sql1} 
    and  submittime <= '2024-06-01 00:00:00' and submittime > '2024-01-01 00:00:00'
'''


print(sql1_1)


record_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

print(record_time)