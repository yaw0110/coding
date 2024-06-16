import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
train = pd.read_csv("flight-delays-fall-2018/flight_delays_train.csv.zip", compression='zip')
test = pd.read_csv("flight-delays-fall-2018/flight_delays_test.csv.zip", compression='zip')
train.head()
test.head()
print(train.info())
print(test.info())
train.describe()
all_data = pd.concat([train, test], ignore_index=True)
all_data.sample(15)
# change target name to make it easier
train = train.rename(columns={'dep_delayed_15min':'delayed'})
all_data = all_data.rename(columns={'dep_delayed_15min':'delayed'})
# change target to numerical N-->0 & Y-->1
train.loc[(train.delayed == 'N'), 'delayed'] = 0
train.loc[(train.delayed == 'Y'), 'delayed'] = 1
all_data.loc[(all_data.delayed == 'N'), 'delayed'] = 0
all_data.loc[(all_data.delayed == 'Y'), 'delayed'] = 1
train['DayofMonth'] = train['DayofMonth'].str.split('-').str[1]
train['Month'] = train['Month'].str.split('-').str[1]
train['DayOfWeek'] = train['DayOfWeek'].str.split('-').str[1]

all_data['DayofMonth'] = all_data['DayofMonth'].str.split('-').str[1]
all_data['Month'] = all_data['Month'].str.split('-').str[1]
all_data['DayOfWeek'] = all_data['DayOfWeek'].str.split('-').str[1]
all_data
order = range(1, 13)
fig , ax = plt.subplots(1, 2, figsize=(8,2))
sns.countplot(data=train, x='Month', order=order, ax=ax[0])
ax[0].set_title('Nb of flights by month')
sns.countplot(data=train, x='Month', hue='delayed', order=order, ax=ax[1])
ax[1].set_title('Delayed/Not delayed flights by month')
plt.figure(figsize=(8,2))
sns.barplot(data=train, x = 'Month', y = 'delayed',order=order )

plt.show()
我们可以看到，所有月份的航班数量和延误数量几乎相同。不过，六月、七月和十二月的延迟率略高，可能是由于假期原因。
order = range(1, 32)

fig, ax = plt.subplots(3, 1, figsize=(8,8))
sns.countplot(x='DayofMonth', data=train, ax=ax[0],order=order)
ax[0].set_title('Nb of flights by day of month')
sns.countplot(x='DayofMonth', hue='delayed', data=train, ax=ax[1],order=order)
ax[1].set_title('Delayed/not Delayed flight by day of month')
sns.barplot(x='DayofMonth', y='delayed', data=train, ax=ax[2], order=order)
ax[2].set_title('Rate of delayed flights by day of month')

plt.tight_layout()
plt.show()
同样，很难说每个月的日子之间是否存在很大差异但是，我们可以说，在该月的最后几天，延迟率较高
order = range(1,8)

fig, ax = plt.subplots(1, 3, figsize=(11,3))
sns.countplot(x='DayOfWeek', data=train, ax=ax[0],order=order)
ax[0].set_title('Nb of flights by day of week')
sns.countplot(x='DayOfWeek', hue='delayed', data=train, ax=ax[1],order=order)
ax[1].set_title('Delayed or not flights by day of week')
sns.barplot(x='DayOfWeek', y='delayed', data=train, ax=ax[2],order=order)
ax[2].set_title('Rate of delayed flights by day of week')

plt.tight_layout()
plt.show()
在这里我们可以看到，周四和周五的航班延误率最高，而周二、周三和周六的航班延误率最低
plt.hist(train.DepTime)
plt.xlabel('Departure Time')
由于值范围很大，一旦我们对它进行分类，我们就会回到这个变量
fig, ax = plt.subplots(3, 1, figsize=(8,8))
sns.countplot(x='UniqueCarrier', data=train, ax=ax[0])
ax[0].set_title('Nb of flights per unique carrier')
sns.countplot(x='UniqueCarrier', hue='delayed', data=train, ax=ax[1])
ax[1].set_title('Nb of delayed/not flights by unique carrier')
sns.barplot(x='UniqueCarrier',y= 'delayed', data=train, ax=ax[2])
ax[2].set_title('Rate of delayed flights by unique carrier')

plt.tight_layout()
plt.show()
我们可以看到UniqueCarrier变量对于延迟有很好的作用
# 指定更多的分箱数量
plt.hist(train.Distance, bins=100)
plt.xlabel('Distance')
plt.show()
我们可以看到，，大多数航班的距离都很短，不到1000英里，标准化和/或缩放此变量是个好主意吗？或者这样差异是否更有意义？也许bin这个变量
all_data.columns
all_data['Month'] = all_data['Month'].astype(int)
all_data['DayofMonth'] = all_data['DayofMonth'].astype(int)
all_data['DayOfWeek'] = all_data['DayOfWeek'].astype(int)
# 确保其他布尔列已经被转换为0和1
all_data.replace(to_replace=[False, True], value=[0, 1], inplace=True)
all_data
all_data['flight'] = all_data['Origin'] + '->' + all_data['Dest']
from sklearn.preprocessing import LabelEncoder

# 将分类变量编码为数值变量
label_encoder = LabelEncoder()
all_data['UniqueCarrier'] = label_encoder.fit_transform(all_data['UniqueCarrier'])
all_data['Origin'] = label_encoder.fit_transform(all_data['Origin'])
all_data['Dest'] = label_encoder.fit_transform(all_data['Dest'])

all_data['flight'] = label_encoder.fit_transform(all_data['flight'])
all_data
from sklearn.preprocessing import StandardScaler

# 初始化StandardScaler对象
scaler = StandardScaler()

# 提取distance和deptime列的数据，创建一个新数据框
to_scale = all_data[['Distance', 'DepTime']]

# 使用scaler对象对这两列数据进行标准化
scaled_data = scaler.fit_transform(to_scale)

# 将标准化后的数据转换回DataFrame并替换原有的列
all_data[['Distance', 'DepTime']] = pd.DataFrame(scaled_data, columns=['Distance', 'DepTime'])

# 检查标准化后的数据
print(all_data.head())
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from scipy import stats
from scipy.stats import norm, skew
plt.style.use('fivethirtyeight')

def draw_dist_prob(data):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(24, 12), dpi=300)

    for i,j in enumerate(['Distance', 'DepTime']):
        sns.distplot(data[j], fit=norm, ax=ax[0][i])
        (mu, sigma) = norm.fit(data[j])
        ax[0][i].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
        ax[0][i].set_ylabel('数量')
        ax[0][i].set_title('{} 频数图'.format(j))

        stats.probplot(data[j], plot=ax[1][i])

draw_dist_prob(all_data)
new_train = all_data.iloc[:100000]
new_test = all_data.iloc[100000:]
pd.DataFrame([i for i in zip(new_train.columns,new_train.skew(),new_train.kurt())],
             columns=['特征','偏度','峰度'])
print(new_train.columns)
new_train
X = new_train.drop(columns=['delayed'])  # 特征
y = new_train['delayed']  # 目标变量

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=48)

print("训练集样本数：", len(X_train))
print("测试集样本数：", len(X_test))
# 假设Graphviz的可执行文件路径是 /path/to/graphviz/bin
graphviz_path = r'C:/_program/Graphviz2.38/bin'

# 设置环境变量
import os
os.environ['PATH'] += os.pathsep + graphviz_path
## 逻辑回归
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
estimator = LogisticRegression(random_state=48)
estimator.fit(X_train, y_train)
coefficients = estimator.coef_
intercept = estimator.intercept_

print(f'系数：{coefficients}')
print(f'截距：{intercept}')
from sklearn.metrics import recall_score

y_pred = estimator.predict(X_test)
print("Accuracy:", y_pred)

score = estimator.score(X_test, y_test)
print("Score:", score)

recall = recall_score(y_test, y_pred, average='macro')
print('Recall:', recall)
feature_names = X_train.columns
weights = estimator.coef_[0]

# 绘制特征权重
import pandas as pd
import seaborn as sns

# 创建一个 DataFrame 来可视化
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Weight': weights
})

# 绘制条形图
sns.barplot(x='Weight', y='Feature', data=feature_importance_df)
plt.show()
print(estimator.get_params())
y_test = np.where(y_test > 0.5, 1, 0)
roc_auc_score(y_test, y_pred)
## xgboost
from xgboost import XGBClassifier

XGBR_classifier = XGBClassifier(random_state=48)
XGBR_classifier.fit(X_train, y_train)
import xgboost as xgb
import matplotlib.pyplot as plt

y_pred = XGBR_classifier.predict(X_test)
xgb.plot_importance(XGBR_classifier, importance_type='gain')
plt.show()
plt.figure(figsize=(20, 10))
xgb.plot_tree(XGBR_classifier, num_trees=48)
plt.savefig("model/xgb.png", dpi=3000)  # 保存为 DPI 为 300 的图像
plt.close()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score
import matplotlib
XGBR_classifier.fit(X_train, y_train)
xgb_y_pred = XGBR_classifier.predict(X_test)
print('xgboost混淆矩阵:',confusion_matrix(y_test,xgb_y_pred))
print('xgboostf1得分:',f1_score(y_test,xgb_y_pred))
rf_accuracy = accuracy_score(y_test, xgb_y_pred)
print("xgboost准确率：", rf_accuracy)
xgb_recall = recall_score(y_test, xgb_y_pred, average='macro')
print("xgboost 召回率：", xgb_recall)
## 决策树
# 创建决策树模型
dtree = DecisionTreeClassifier()

# 训练模型
dtree.fit(X_train, y_train)
dt_y_pred = dtree.predict(X_test)
print('决策树混淆矩阵:',confusion_matrix(y_test,dt_y_pred))
print('决策树f1得分:',f1_score(y_test,dt_y_pred))
dt_accuracy = accuracy_score(y_test, dt_y_pred)
print("决策树准确率：", dt_accuracy)
dt_recall = recall_score(y_test, dt_y_pred, average='macro')
print("决策树召回率：", dt_recall)
import joblib
joblib.dump(dtree,'model/dt_classifier.pkl')
from sklearn import tree
import graphviz

# 假设 dtree 是已经训练好的决策树模型

# 导出决策树为 dot 格式
dot_data = tree.export_graphviz(dtree,
                                 feature_names=X_train.columns.tolist(),
                                 class_names=np.unique(y_train).astype(str).tolist(),
                                 filled=True, rounded=True,
                                 special_characters=True)

# 将 dot 数据写入文件
with open("tree.dot", "w") as f:
    f.write(dot_data)

# 使用 Graphviz 的 dot 命令行工具来生成图像
# 您可以在命令行中运行以下命令来生成高分辨率的图像
# 例如，生成 DPI 为 300 的 PNG 图像：
os.system('dot -Tpng -o output_highres.png tree.dot -Gdpi=1300')
from sklearn.tree import export_graphviz
# 可视化随机森林中的第一棵树
dot_data = export_graphviz(dtree,
                                 feature_names=X_train.columns.tolist(),
                                 class_names=np.unique(y_train).astype(str).tolist(),
                                 filled=True, rounded=True,
                                 special_characters=True)

# 使用 graphviz 库将 DOT 数据转换为可视化图形
graph = graphviz.Source(dot_data)
graph.render("dtree")  # 会创建一个名为 "rf_tree.pdf" 的文件


from IPython.display import IFrame
IFrame('dtree.pdf', width='100%', height=400)
## 随机森林
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import  cross_val_score

# trainErr = []
# testErr = []
# CVErr = []
# K = np.arange(2, 15)

# for k in K:
#     modelRF = RandomForestRegressor(n_estimators=k, random_state=48)
#     modelRF.fit(X_train, y_train)
#     trainErr.append(1 - modelRF.score(X_train, y_train))
#     testErr.append(1 - modelRF.score(X_test, y_test))
#     Err = 1 - cross_val_score(modelRF, X, y, cv=5, scoring='r2')
#     CVErr.append(Err.mean())

# fig = plt.figure(figsize=(15, 6))
# ax1 = fig.add_subplot(121)
# ax1.grid(True, linestyle='-.')
# ax1.plot(K, trainErr, label="训练误差", marker='o', linestyle='-')
# ax1.plot(K, testErr, label="旁置法测试误差", marker='o', linestyle='-')
# ax1.plot(K, CVErr, label="5-折交叉验证误差", marker='o', linestyle='--')
# ax1.set_xlabel("树的数量")
# ax1.set_ylabel("误差（1-R方）")
# ax1.set_title('树的数量和误差')
# ax1.legend()

# plt.show()
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(random_state=48)

rf_classifier.fit(X_train, y_train)
rf_y_pred = rf_classifier.predict(X_test)

print('随机森林混淆矩阵：', confusion_matrix(y_test, rf_y_pred))
print('随机森林F1得分：', f1_score(y_test, rf_y_pred))
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print("随机森林准确率：", rf_accuracy)

rf_recall = recall_score(y_test, rf_y_pred, average='macro')
print("随机森林召回率：", rf_recall)
from sklearn.tree import export_graphviz
# 可视化随机森林中的第一棵树
dot_data = export_graphviz(rf_classifier.estimators_[0], out_file=None,
                           feature_names=X_train.columns,  # 假设 X_train.columns 包含了特征名称
                           class_names=rf_classifier.classes_.astype(str).tolist(),
                           filled=True, rounded=True, special_characters=True)

# 使用 graphviz 库将 DOT 数据转换为可视化图形
graph = graphviz.Source(dot_data)
graph.render("rf_tree")  # 会创建一个名为 "rf_tree.pdf" 的文件


from IPython.display import IFrame
IFrame('rf_tree.pdf', width='100%', height=400)
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

# 创建 CatBoost 模型
catboost_model = CatBoostClassifier(random_state=48)

# 准备数据，CatBoost 需要一个 Pool 对象
train_data = Pool(data=X, label=y)
test_data = Pool(data=X_test, label=y_test)

# 训练模型
catboost_model.fit(train_data, eval_set=test_data, plot=True)

# 预测
y_pred = catboost_model.predict(X_test)
import catboost
import matplotlib.pyplot as plt

feature_importances = catboost_model.feature_importances_

import matplotlib.pyplot as plt

# 获取特征名称
feature_names = new_train.columns

# 绘制特征重要性
plt.barh(range(len(feature_importances)), feature_importances, color='blue')
plt.yticks(range(len(feature_importances)), feature_names)
plt.show()
new_train.columns
# 评估模型
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
print('CatBoost 混淆矩阵:',confusion_matrix(y_test,y_pred))
print('CatBoost f1得分:',f1_score(y_test,y_pred))
catboost_accuracy = accuracy_score(y_test, y_pred)
print("CatBoost 准确率：", catboost_accuracy)
print("CatBoost 召回率", recall_score(y_test, y_pred, average='macro'))
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV

# 假设X_train, y_train是训练数据和标签
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=48)

# 定义超参数搜索空间
param_space = {
    'learning_rate': (0.01, 1.0, 'log-uniform'),
    'depth': (3, 5),
    'l2_leaf_reg': (1, 3),
}

# 初始化CatBoost模型
model = CatBoostClassifier(random_seed=48)

# 使用BayesSearchCV进行超参数优化
bayes_search = BayesSearchCV(
    estimator=model,
    search_spaces=param_space,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    verbose=0
)

# 拟合贝叶斯优化模型
bayes_search.fit(X_train, y_train, eval_set=(X_val, y_val))

# 最佳参数
best_params = bayes_search.best_params_
print("Best parameters found: ", best_params)

# 使用最佳参数训练模型
best_model = CatBoostClassifier(**best_params)
best_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20)
best_model.get_feature_importance()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 假设X_val, y_val是验证集数据和标签
y_pred_val = best_model.predict(X_val)

# 评估指标
accuracy = accuracy_score(y_val, y_pred_val)
precision = precision_score(y_val, y_pred_val)
recall = recall_score(y_val, y_pred_val, average='macro')
f1 = f1_score(y_val, y_pred_val)
roc_auc = roc_auc_score(y_val, y_pred_val)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC-ROC: {roc_auc}")
sample = pd.read_csv("flight-delays-fall-2018/sample_submission.csv.zip", compression='zip')
sample.head(900)
predictions = catboost_model.predict_proba(new_test)[:, 1]

submission = pd.DataFrame({'id':range(100000),'dep_delayed_15min':predictions})
submission.head(900)
filename = 'flight_delay.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)
import joblib
joblib.dump(catboost_model,'model/catboost_classifier.pkl')
