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
# 첫번째 행부터 5개의 행을 출력하기
csv_header = first_data.head(n=5)
# 광고비와 매출액에 대한 정보를 출력을 한다.
print(f"header : \n{csv_header}")

# 데이터 전처리
target = 'Sales'
# x, y 나누기
x = first_data.drop(target, axis=1)
print(f"x data : \n{x}")
y = first_data.loc[:, target]
print(f"y data : \n{y}")

# 학습 데이터와 검증 데이터로 분리 -> 학습을 위한 데이터와 검증을 위한 데이터로 분리를 한다.
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.3, random_state=20)

# 데이터 크기 조정
scaler = MinMaxScaler()
x_train_s = scaler.fit_transform(x_train)
x_val_s = scaler.transform(x_val)

# 모델링 만들기 -> LinearRegression (선형 회귀 모델을 이용하기 위해서 사용)
linear_model = LinearRegression()

# 모델 데이터 학습
linear_model.fit(x_train_s, y_train)

# 모델에 대한 검증
pred = linear_model.predict(x_val_s)

# 모델에 대한 검증에 대한 평가를 진행한다.
# 절대 값 오차의 평균, 평균 오차에 대한 검증을 한다.
predict_error = mean_absolute_error(y_val, pred)
print(f"error : \n{predict_error}")
