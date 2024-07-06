import os

# 发现文件名由学号+姓名组成，所以用正则表达式提取学号列和姓名列
import re
import pandas as pd



def get_data(filename):
    def get_number(filename):
        pattern = re.compile(r'\d+')
        number = re.findall(pattern, filename)
        return number[0]
    def get_name(filename):
        pattern = re.compile(r'\d+(.*?)\.pdf')
        name = re.findall(pattern, filename)
        return name[0].strip()
    
    number = get_number(filename)
    name = get_name(filename)
    return number, name


path = r'C:\Users\yaw\Documents\21数管2留宿名单\21数管2留宿名单'

filename_list = []
excel_list = []
# 遍历该文件夹，获取文件名
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.pdf'):  # 筛选
            filename_list.append(file)
        else :
            excel_list.append(file)

data = []
for filename in filename_list:
    number, name = get_data(filename)
    data.append([number, name])
df = pd.DataFrame(data, columns=['学号', '姓 名'])
df.to_csv('data.csv', index=False)


# 读取xlsx文件的姓名、楼栋、宿舍号、留校事由（请如实填写，例如：实习；找实习或找工作；备考等)
excel_df = pd.read_excel(path + '\\' + excel_list[0], skiprows=1)[['姓 名', '楼栋', '宿舍号','留校事由（请如实填写，例如：实习；找实习或找工作；备考等）']].dropna()


new_df = excel_df.join(df.set_index('姓 名'), on='姓 名')
new_df['宿舍号'] = new_df['宿舍号'].apply(lambda x: str(x).replace('.0', ''))
new_df['目前所在宿舍'] = new_df['楼栋'] + new_df['宿舍号']

new_df['留校原因'] = new_df['留校事由（请如实填写，例如：实习；找实习或找工作；备考等）']
new_df['备注'] = ''


print(new_df[['学号', '姓 名', '目前所在宿舍', '留校原因', '备注']])

# 编码为utf-8，。写入csv文件中

new_df[['学号', '姓 名', '目前所在宿舍', '留校原因', '备注']].to_excel(r'C:\Users\yaw\Desktop\name.xlsx', index=False)