import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split  # 데이터 분할
from sklearn.metrics import *  # 평가에 대한 패키지
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # scaler 패키기 추가
from sklearn.linear_model import LinearRegression  # 만들 모델에 대한 패키지 추가
from sklearn.tree import DecisionTreeClassifier  # 분류 모델에 대한 패키지 추가

path = "https://raw.githubusercontent.com/DA4BAM/dataset/master/Graduate_apply.csv"

# 데이터를 읽어오기
learning_data = pd.read_csv(path)
data_header = learning_data.head()

# 데이터 분할
target = 'admit'  # 합격 및 탈락에 대한 표시가 되어 있는 값 0은 탈락 1은 합격이다.
x = learning_data.drop(target, axis=1)
y = learning_data.loc[:, target]

# 가변수화 -> 해당 데이터에 대한 값에 대한 변경
# 가변수화는 범주형 정보들을 숫자로 바꿔주는 방법이다.
# 가변수화는 반드시 수행해 줘야 될 전처리 단계 이다.
cat_cols = ['rank']
# drop_first는 첫번쨰 컬럼에 대한 값을 제거 하는 것
x = pd.get_dummies(x, columns=cat_cols, drop_first=True)

# 데이터 분할
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=20)

# 데이터 스케일링
scaler = MinMaxScaler()
x_train_s = scaler.fit_transform(x_train)
x_val_s = scaler.transform(x_val)

# 모델링
# 모델 선언
decision_model = DecisionTreeClassifier()

# 학습
decision_model.fit(x_train_s, y_train)

# 검증 -> 예측
pred = decision_model.predict(x_val_s)
print(f"predicted : \n{pred}")

# 검증 -> 평가
score_accuracy = accuracy_score(y_val, pred)
print(f"Accuracy : {score_accuracy}")
