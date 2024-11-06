import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam


# dataset을 만들어주는 함수 -> tensor 형태의 데이터 생성
def make_dataset(x_train, x_val, y_train, y_val, batch_size=32) -> []:
    # tensor 변환
    x_train_tensor = torch.tensor(data=x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(data=y_train.values, dtype=torch.float32).view(-1, 1)
    # 평가를 위한 데이터 tensor로 변환
    x_val_tensor = torch.tensor(data=x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(data=y_val.values, dtype=torch.float32).view(-1, 1)

    # 학습을 위한 데이터 합치기
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

    # torch에서 데이터를 가져오는 객체인 DataLoader생성
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=True)  # shuffle은 데이터를 무작위 순서로 섞어주는 것을 말한다.
    # 데이터를 불러오는 객체 반환
    return train_dataloader, x_val_tensor, y_val_tensor


# 학습을 진행을 하는 함수
def train(dataloader, model, loss_fn, optimizer, device) -> float:
    # 전체 데이터 셋의 크기
    size = len(dataloader.dataset)
    # batch 크기
    num_batches = len(dataloader)
    # 오차율을 저장할 변수
    tr_loss = 0
    # model에 대한 모드 설정
    model.train()
    # batch 크기 만큼 반복하면서 학습
    for batch, (X, y) in enumerate(dataloader):
        # 입력 데이터와 레이블을 지정된 장치로 이동
        X, y = X.to(device), y.to(device)
        # 모델로 예측
        pred = model(X)
        # 오차 함수를 이용해서 저압과 오차율 확인
        loss = loss_fn(pred, y)
        # 오차에 대한 값을 더해서 저장
        tr_loss += loss

        # model에 대한 가중치 및 편차 변경
        # 역전파를 통해 모델의 각 파라미터에 대한 손실의 기울기를 계산
        loss.backward()
        # 옵티마이저가 계산된 기울기를 상용하여 모델의 파라미터를 변경
        optimizer.step()
        # 옵티마이저의 기울기 값 초기화 -> 기울기가 누적되는 것을 방지한다.
        optimizer.zero_grad()
    # 모드 배치에 대한 오차율 평균 구하기
    tr_loss /= num_batches

    return tr_loss.item()


# 평가를 하기 위한 함수
def evaluate(x_val_tensor, y_val_tensor, model, loss_fn, device) -> []:
    # 모델을 평가 모드로 변경
    model.eval()
    # 평가를 하는 과정에서 기울기를 계산하지 않도록 설정 -> 메모리 사용을 줄이고 평가 속도를 높인다.
    with torch.no_grad():
        # 평가 데이터와 레이블을 지정한 장치로 이도
        x, y = x_val_tensor.to(device), y_val_tensor.to(device)
        # 모델을 통한 예측
        pred = model(x)
        # 평가에 대한 오차 값
        eval_loss = loss_fn(pred, y).item()
    return eval_loss, pred


# 그래프로 보여주기 위한 함수
def dl_learning_curve(tr_loss_list, val_loss_list) -> None:
    # 학습률
    epochs = list(range(1, len(tr_loss_list) + 1))

    plt.plot(epochs, tr_loss_list, label='train_err', marker='.')
    plt.plot(epochs, val_loss_list, label='val_err', marker='.')

    plt.ylabel(ylabel='Loss')
    plt.xlabel(xlabel='Epoch')
    plt.legend()
    plt.grid()
    plt.show()


# torch에 대한 학습을 위한 모드 설정 -> CPU, GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

# 데이터 로딩
path = 'https://raw.githubusercontent.com/DA4BAM/dataset/master/Carseats.csv'
# pandas 형태로 데이터 읽기
data = pd.read_csv(path)
# 가져온 파일에 대한 정의 값 확인
header = data.head()
print(f"header value : {header}")

# 필요한 데이터 준비하기
# 필요 없는 데이터 삭제
target = 'Sales'
# 타겟에 대한 데이터의 열 삭제
x = data.drop(target, axis=1)
# 학습 시 확인할 값에 대한 분리 -> 정답에 대한 값
y = data.loc[:, target]

# 범주형 변수에 대해서 가변수화 진행
cat_cols = ['ShelveLoc', 'Education', 'Urban', 'US']
x = pd.get_dummies(x, columns=cat_cols, drop_first=True)

# 데이터에 대한 분할
# 학습과 검증을 위한 비율을 8 : 2로 설정하기 위해서 test_size를 0.2로 정의 한다.
# 결과를 같게 나오기 위해서 사용이 되는 값이 random_state이다. -> 랜덤 시드 값이라고 이야기를 한다.
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2, random_state=20)
# scaler를 이용한 데이터 변경
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# 텐서 데이터로 변경
train_loader, x_val_ts, y_val_ts = make_dataset(x_train, x_val, y_train, y_val, 32)

for x, y in train_loader:
    print(f"Shape of x [rows, columns] : {x.shape}")
    print(f"Shape of y : {y.shape} {y.dtype}")
    break

n_feature = x.shape[1]

print(f"feature : {n_feature}")

# 모델에 대한 구조 및 설계
model1 = nn.Sequential(
    nn.Linear(n_feature, 8),  # 입력 파라미터와 출력 파라미터 정의 -> 션형 레이어로 설정
    nn.ReLU(),  # 활성 함수에 대한 정의
    nn.Linear(8, 4),  # 입력 파라미터와 출력 파라미터 정의 -> 션형 레이어로 설정
    nn.ReLU(),  # 활성 함수에 대한 정의
    nn.Linear(4, 1),  # 입력 파라미터와 출력 파라미터 정의 -> 션형 레이어로 설정
).to(device)

print(model1)

# 손실 함수에 대한 정의 -> 회귀 모델에 대한 오차 함수는 MSELoss()이다.
loss_fn = nn.MSELoss()
# 최적화 함수에 대한 정의
optimizer = Adam(model1.parameters(), lr=0.001)

epochs = 20
tr_loss_list, val_loss_list = [], []

for t in range(epochs):
    tr_loss = train(train_loader, model1, loss_fn, optimizer, device)
    val_loss, _ = evaluate(x_val_ts, y_val_ts, model1, loss_fn, device)

    # 리스트에 loss 추가 --> learning curve 그리기 위해.
    tr_loss_list.append(tr_loss)
    val_loss_list.append(val_loss)

    print(f"Epoch {t + 1}, train loss : {tr_loss:4f}, val loss : {val_loss:4f} ")

# 그래프로 학습에 대한 오차에 대한 값 배열에 담기
dl_learning_curve(tr_loss_list, val_loss_list)

# 평가에 대한 오차와 평가 값 담기
loss, pred = evaluate(x_val_ts, y_val_ts, model1, loss_fn, device)

mae = mean_absolute_error(y_val_ts.numpy(), pred.numpy())
mape = mean_absolute_percentage_error(y_val_ts.numpy(), pred.numpy())

print(f'MSE : {loss}')
print(f'MAE : {mae}')
print(f'MAPE : {mape}')
