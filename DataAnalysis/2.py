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
# 创建一个图形和两个子图
fig, axs = plt.subplots(1, 2, figsize=(14, 5))  # 1行2列

sns.boxplot(data=train, x='Distance', ax=axs[0])
axs[0].set_title('Boxplot for Distance')

sns.boxplot(data=train, x='DepTime', ax=axs[1])
axs[1].set_title('Boxplot for DepTime')

plt.tight_layout()
plt.show()
