# -*- coding:UTF-8 -*-
import os  # 操作系统的功能函数
import tarfile  # 压缩包相关功能函数
from six.moves import urllib  # url标准访问库，six=2*3 ，屏蔽python2和3带来的url差异
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Imputer

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    # 解压
    tgz_path = os.path.join(housing_path, "housing.tgz")
    housing_tgz = tarfile.open(tgz_path, 'r')
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    # 读取
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# 抓取与读取数据
# fetch_housing_data()
housing = load_housing_data()

# 查看数据集的基本信息&绘图
pd.set_option('display.max_columns', None)
print(len(housing))
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
pprint.pprint(housing.describe())
housing.hist(bins=60, figsize=(20, 15))
plt.show()
print(np.random.permutation(len(housing)))


# 分数据集和测试集


def split_train_test_self(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))  # 生成一个大小等于数据集大小的乱序的序列（数字）
    test_set_size = int(len(data) * test_ratio)  # 生成测试集数据个数
    test_indices = shuffled_indices[:test_set_size]  # 取与测试集数量n相同的乱序数，相当于在shuffled——indices中取前n个数
    train_indices = shuffled_indices[test_set_size:]  # 取剩余的训练集乱序数，相当于取n后面的数
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = split_train_test_self(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")

train_set2, test_set2 = train_test_split(housing, test_size=0.2,
                                         random_state=42)  # 直接调用sklearn.modelselection包中的函数，random_state
# 是随机种子，保证每次随机划分都是一样的

# 因为收入中位数是一个重要的属性，所以希望在确保收入属性的基础上，测试集能覆盖数据集中各种不同类型的收入
# 因为收入中位数是一个连续的数值属性，所以可以先创建一个收入类别的属性
# 因为不能分层太多，保证每层的数据量，所以收入中位数除以1.5，然后使用ceil向上取整
# 通过直方图观察到，收入中位数大多位于2-5，所以将所有大于5的类别合并为类别5

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

# 现在可以根据收入类别进行分类了

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)  # 创建一个split实例
# 使用split这个实例，调用spilit方法，第一个参数为数据集，第二个为按照比例划分的特征
for train_index, test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# 查看一下结果是否符合预期，测试各个收入类别的占比
print(housing["income_cat"].value_counts() /len(housing))
print(strat_train_set.head())

# 在利用完income_cat完成分类之后，就可以删除该属性了

for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

# 数据探索与可视化
# 进行一个训练集样本复制，避免对训练集的损失，如果样本过大也可以选择用抽样

housing = strat_test_set.copy()

# 可视化地理信息
# 参数：kind：点的种类；x：x坐标轴数据，y：y坐标轴数据，alpha：透明度，越密集越清晰
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()

# 绘制房价的信息
# 大小圈绘图

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100,label="population",
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
plt.show()

# 寻找相关性
# 计算数值
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
print(corr_matrix["median_income"].sort_values(ascending=False))

# 使用pandas的plot绘制相关性的图
attributes = ["median_house_value","median_income", "total_rooms","housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))

#发现收入中位数和房价中位数相关性比较好，单独绘制进行观察
housing.plot(kind="scatter", x="median_income", y="median_house_value",alpha=0.1)
plt.show()

# 进行特征组合，比如把房间总数变成人均房间数，把卧室数量变成卧室占总房间的比，以及每个家庭的人数
# 通过组合特征，然后在通过相关性矩阵分析看是否有产生更好的特征
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing['population'] / housing["households"]
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))


# 将数据和标签分离开，drop会创建一个副本，不会动原本的数据
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# 数据清理
# 大部分机器学习无法在缺失的数据机上进行工作，我们要创建一些辅助函数来辅助它
# 有三种选择：放弃相应的地区，放弃这个属性，将确实值设为0（平均数或者中位数）
# 通过DataFrame的dropna(),drop(),fillna()可以轻松的完成

housing.dropna(subset=["total_bedrooms"]) #  option 1
housing.drop("total bedrooms", axis=1) # option 2
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median) #option 3: 可以尝试替换缺失值，来改进性能

# Sklearn 提供了一个imputer进行缺失值处理，使用方法如下：
imputer = Imputer(strategy="median")
#由于中位数只能进行数值属性计算，所以我们需要创建一个没有文本属性的数据副本


