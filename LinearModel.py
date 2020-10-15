#-*- coding:UTF-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import sklearn

#加载数据
oecd_bli = pd.read_csv("dataset/lifesat/oecd_bli_2015.csv",thousands=',')
gdp_per_capita = pd.read_csv("dataset/lifesat/gdp_per_capita.csv",thousands=',',delimiter='\t',encoding='latin1',na_values="n/a")

#测试读取的数据
print(oecd_bli.head(10))


