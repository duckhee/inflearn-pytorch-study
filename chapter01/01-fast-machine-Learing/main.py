import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split  # 데이터 분할
from sklearn.metrics import *  # 평가에 대한 패키지
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # scaler 패키기 추가
from sklearn.linear_model import LinearRegression  # 만들 모델에 대한 패키지 추가
from sklearn.tree import DecisionTreeClassifier  # 분류 모델에 대한 패키지 추가

# 데이터를 가져올 URL
path = 'https://raw.githubusercontent.com/DA4BAM/dataset/master/advertising.csv'

# data pandas로 읽어와서 객체 생성
first_data = pd.read_csv(path)
# 첫번째 행 읽기
first_data.head()

# 데이터 전처리
target = 'Sales'

x = first_data.drop(target, axis=1)
y = first_data.loc[:, target]
# 학습 데이터와 검증 데이터로 분리
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.3, random_state=20)

# 데이터 크기 조정
scaler = MinMaxScaler()
x_train_s = scaler.fit_transform(x_train)
x_val_s = scaler.transform(x_val)

# 모델링 만들기 -> LinearRegression
