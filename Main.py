# -*- coding:UTF-8 -*-

# 预处理相关引用
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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

# 模型相关引用
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 性能验证
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

#模型保存
from sklearn.externals import joblib

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
#print(len(housing))
#print(housing.head())
#print(housing.info())
#print(housing["ocean_proximity"].value_counts())
#pprint.pprint(housing.describe())
housing.hist(bins=60, figsize=(20, 15))
# plt.show()
#print(np.random.permutation(len(housing)))


# 分数据集和测试集


def split_train_test_self(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))  # 生成一个大小等于数据集大小的乱序的序列（数字）
    test_set_size = int(len(data) * test_ratio)  # 生成测试集数据个数
    test_indices = shuffled_indices[:test_set_size]  # 取与测试集数量n相同的乱序数，相当于在shuffled——indices中取前n个数
    train_indices = shuffled_indices[test_set_size:]  # 取剩余的训练集乱序数，相当于取n后面的数
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = split_train_test_self(housing, 0.2)
#print(len(train_set), "train +", len(test_set), "test")

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
#print(housing["income_cat"].value_counts() /len(housing))
#print(strat_train_set.head())

# 在利用完income_cat完成分类之后，就可以删除该属性了

for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

# 数据探索与可视化
# 进行一个训练集样本复制，避免对训练集的损失，如果样本过大也可以选择用抽样

housing = strat_test_set.copy()

# 可视化地理信息
# 参数：kind：点的种类；x：x坐标轴数据，y：y坐标轴数据，alpha：透明度，越密集越清晰
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show()

# 绘制房价的信息
# 大小圈绘图

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100,label="population",
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
# plt.show()

# 寻找相关性
# 计算数值
corr_matrix = housing.corr()
#print(corr_matrix["median_house_value"].sort_values(ascending=False))
#print(corr_matrix["median_income"].sort_values(ascending=False))

# 使用pandas的plot绘制相关性的图
attributes = ["median_house_value","median_income", "total_rooms","housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))

#发现收入中位数和房价中位数相关性比较好，单独绘制进行观察
housing.plot(kind="scatter", x="median_income", y="median_house_value",alpha=0.1)
# plt.show()

# 进行特征组合，比如把房间总数变成人均房间数，把卧室数量变成卧室占总房间的比，以及每个家庭的人数
# 通过组合特征，然后在通过相关性矩阵分析看是否有产生更好的特征
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing['population'] / housing["households"]
corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))


# 将数据和标签分离开，drop会创建一个副本，不会动原本的数据
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# 数据清理
# 大部分机器学习无法在缺失的数据机上进行工作，我们要创建一些辅助函数来辅助它
# 有三种选择：放弃相应的地区，放弃这个属性，将确实值设为0（平均数或者中位数）
# 通过DataFrame的dropna(),drop(),fillna()可以轻松的完成

#housing.dropna(subset=["total_bedrooms"]) #  option 1
#housing.drop("total bedrooms", axis=1) # option 2
#median = housing["total_bedrooms"].median()
#housing["total_bedrooms"].fillna(median) #option 3: 可以尝试替换缺失值，来改进性能

#  Sklearn 提供了一个imputer进行缺失值处理，使用方法如下：
imputer = Imputer(strategy="median")
#  由于中位数只能进行数值属性计算，所以我们需要创建一个没有文本属性的数据副本，也就是去除ocean_proximity属性
housing_num = housing.drop("ocean_proximity", axis=1)
#  使用fit方法将imputer实例适配到训练集：imputer仅仅是计算每个属性的中位数值，
imputer.fit(housing_num)
#  用训练后的imputer完成空缺值的转化,转化后是一个Numpy数组，我们可以把它进一步转化为DataFrame
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
#print(housing_num.columns)

#  文本处理与分类属性
#  之前对ocean_proximity直接采取了drop，因为机器学习算法对数字更易打交道，所以我们需要对文本进行编码
#  Sklean为这类任务提供了一个转化起LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded_DF = pd.DataFrame(housing_cat_encoded,columns=['housing_cat'])
#print(encoder.classes_)
# type()分析object的类型：type(housing_encoded)
# 使用encoder.classes_可以查看编码对应的类型

# Sklearn同样提供一个OnehotEncoder用来将整数分类值转化为独热向量。
# 但是fit_transform()需要的是一个二维数组，但是我们之前如果用LabelEncoder生成的是一维的数组，
# 所以对一维的数组记得要进行使用reshape()重塑。
# onehot产生的是一个稀疏矩阵，可能产生数千行，每一行都是一个数组，只有一个1，其余全是0，存储浪费，所以只存了1的位置
# 如果想转化成矩阵，可以直接调用toarray()进行转化
housing_cat_encoded_reshaped = housing_cat_encoded.reshape(-1,1)
onehot_encoder = OneHotEncoder()
housing_cat_onehot_encoded = onehot_encoder.fit_transform(housing_cat_encoded_reshaped)

# 使用LabelBinarizer类可以直接完成两个转化，即LabelEncoder和OnehotEncoder
# 此时返回的则是一个Numpy的密集数组，如果想得到onehot的稀疏矩阵，则可以增加sparse_output=True的参数
binarizer_encoder = LabelBinarizer()
housing_cat_binarizer_encoder = binarizer_encoder.fit_transform(housing_cat)

# 自定义转化器：可以自定义清理操作或者组合特定属性等任务。并且和Sklearn自身的功能无缝衔接，如流水线。
# sklearn依赖于鸭子类型变异（duck typing）不是继承， 所以只需要创建一个类，然后调用fit，transform或者f%t
# 添加transformerMixin作为基类，就可以直接得到fit_transform()方法
# 添加BaseEstimator作为基类，可以得到自动调整超参数的方法（get_params(),set_params())
# 写一个用来添加组合属性的转化器
rooms_ix, bedrooms_ix,population_ix,household_ix = 3,4,5,6 # 自己根据数据集，读出的列数
class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs,add_bedrooms_per_room 不知道会不会存在，默认True
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing to do else
    def transform(self,X,y=None):
        rooms_per_household = X[:,rooms_ix] / X[:,household_ix] # 每个人得到的房价数量
        population_per_household = X[:,population_ix] / X[:,household_ix] # 每个家庭的人数
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix] / X[:,rooms_ix] # 卧室占总房间数的比例
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household,population_per_household] # 注意np.c_按照行进行链接，但是不是函数，不加()而是[]

# attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = True)
# housing_extra_attribs = attr_adder.transform(housing.values)

# 转换流水线
# 许多数据需要以正确的步骤进行转换，Sklearn提供弄了Pipeline来支持这样的转化
# Pipeline构造函数会通过一系列名称/估算器的匹配来定义步骤序列。除了最后一个是估算器之外，前面都必须是转化器。（也就是说，必须有fit_transform()方法）。命名可以随意。
# 当调用流水线的fit时候，会对最后一个除外的所有步骤按顺序调用fit_transform()，最后一个只调用fit()。
# 因此，可以在结束后在调用一次pipeline.transform()，或者直接调用fit_transform()
#
num_pipeline = Pipeline([
    ('imputer',Imputer(strategy='median')),
    ('attribs_adder',CombinedAttributesAdder()),
    ('std_scaler',StandardScaler()),
])
# housing_num_tr = num_pipeline.fit_transform(housing_num)

# 在创建好流水线后，我们还需要LabelBinarizer来处理分类值
# FeatureUnion类：只需要提供一个转换器列表，当transform方法被调用时，会并行运行每个转化器的transform，等待输出，并连接起来返回结果
# 同样fit也是调用每个转换器的fit方法

num_attribs = list(housing_num) # 纯数字的列名
print(num_attribs)
cat_attribs = ["ocean_proximity"] # 分类属性列名（非数字）

# 因为Sklearn没有工具来处理DataFrame，所以需要自己写一个转化器
class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names].values
# 因为版本问题，不能直接使用LabelBinarizer，需要用户自己定义LabelBinarizer类，并继承LabelBinarizer本身，然后复写fit_transform()方法，修改参数变量个数，增加y=None
class PipelineFriendlyLabelBinarizer(LabelBinarizer):
    def fit_transform(self, X, y=None):
        return super(PipelineFriendlyLabelBinarizer, self).fit_transform(X)
# 创建流水线
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)), # 查找提取指定列名的列（数字列）
    ('imputer', Imputer(strategy='median')), # 进行缺失值处理
    ('attribs_adder', CombinedAttributesAdder()), # 组合属性并添加
    ('std_scaler', StandardScaler()), # 进行特征缩放中的标准化
])
cat_pipeline = Pipeline([
    ('selector',DataFrameSelector(cat_attribs)), # 查找提取指定列名的列（文字列）
    ('label_binarizer',PipelineFriendlyLabelBinarizer()) # 进行编码转化
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

# 运行流水线
housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared)

# 选择和训练模型

# 先做一个简单的线性回归模型
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
print("--train has been finished successfully--")

# 用训练集的几个实例测试一下
some_data = housing.iloc[:5]
some_label = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
# print("prediction:\t", lin_reg.predict(some_data_prepared))
# print(some_label)

# 可以根据sklearn的mean_squared_error的函数来测量整个训练集上的回归模型的RMSE
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels,housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("线性回归的RMSE是：", lin_rmse)

# 得到的预测误差达到68628.19819848922，源自于数据拟合不足。
# 可能是因为数据的特征不能很好的提供预测，或者预测的模型不够强大
# 这里我们可以尝试其他模型
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)
housing_treegre_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels,housing_treegre_predictions)
tree_rmse = np.sqrt(tree_mse)
print("决策树的RMSE是：", tree_rmse)

# 我们可以发现决策树得到的RMSE竟然是0.0，所以我们有理由怀疑其发生了过度拟合
# 这里是我们使用交叉验证进行更好的评估
# 使用10折交叉验证
# cross_val_score()的参数：模型，数据，label，分数计算方法，折叠次数
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores) # 因为得到的scores是负数，所以我们需要取正

# 额外创建计算交叉验证结果的函数
def display_scores(scores):
    print(scores)
    print("Mean:", scores.mean()) # 平均值
    print("Standard deviation:", scores.std()) # 标准差

# 决策树的评分
print("Decision Tree")
display_scores(rmse_scores)

# 对比，线性回归的使用K折叠交叉验证的评分
scores_lin_reg = cross_val_score(lin_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores_lin_reg = np.sqrt(-scores_lin_reg)
print("linear regression")
display_scores(rmse_scores_lin_reg)

# 通过对比我们发现，决策树不管是mean还是std都比线性回归表现的要差，确实发生了过拟合。
# 因为没有使用交叉验证时决策树的RMSE是0，而使用了之后得到的性能很差

# 我们再换一个模型进行尝试，使用随机森林回归模型
random_forest_regressor = RandomForestRegressor()
random_forest_regressor.fit(housing_prepared,housing_labels)
random_forest_regressor_predictions = random_forest_regressor.predict(housing_prepared)
random_forest_regressor_mse = mean_squared_error(housing_labels,random_forest_regressor_predictions) # 不使用交叉验证的RMSE
random_forest_regressor_rmse = np.sqrt(random_forest_regressor_mse)
print("不使用交叉验证得到的随机森林的RMSE:", random_forest_regressor_rmse)
# 对比，随机森林的使用K折叠交叉验证的评分
scores_random_forest_regressor = cross_val_score(random_forest_regressor,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores_random_forest_regressor = np.sqrt(-scores_random_forest_regressor)
print("Random forest Regression")
display_scores(rmse_scores_random_forest_regressor)

# 对模型进行保存
joblib.dump(lin_reg,"model/housing_lin_reg_model.pkl")
joblib.dump(tree_reg,"model/hosing_tree_reg_model.pkl")
joblib.dump(random_forest_regressor,"model/housing_random_forest_reg_model.pkl")
# my_lin_reg = joblib.load("housing_lin_reg_model.pkl")