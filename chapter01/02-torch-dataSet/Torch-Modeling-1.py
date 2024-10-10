import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# torch
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam


# 데이터 셋을 생성하는 함수
def make_dataset(x_train, x_val, y_train, y_val, batch_size=32):
    # 데이터 텐서로 변경
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    # TensorDataset 생성 -> 텐서 데이터 셋으로 합치기
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, x_val_tensor, y_val_tensor


# 데이터 학습
def train(dataLoader, model, loss_fn, optimizer, device):
    size = len(dataLoader.dataset)
    num_batches = len(dataLoader)
    # 전체 데잍 셋의 크기
    tr_loss = 0
    # 훈련 모드로 설정
    model.train()
    # 반복을 통한 학습
    # batch : 현재 배치 번호, (X, y): 입력 데이터와 레이블
    for batch, (X, y) in enumerate(dataLoader):
        # 현재 데이터와 레이블을 지정된 장치 (device, CPU 또는 GPU)로 연결
        X, y = X.to(device), y.to(device)

        # Feed Forward
        pred = model(X)
        loss = loss_fn(pred, y)
        tr_loss += loss

        # Back propagation
        # 역전파를 통해 모델의 각 파라미터에 대한 손실의 기울기를 계산
        loss.backward()
        # 옵티마이저가 계산된 기울기를 사용하여 모델의 파라이터를 업데이트
        optimizer.step()
        # 옵티마이저의 기울기 값 초기화
        # 기울기가 누적되는 것 방지
        optimizer.zero_grad()
    # 모든 배치에서의 loss 평균
    tr_loss /= num_batches

    return tr_loss.item()


# 검증을 위한 함수
def evaluate(x_val_tensor, y_val_tensor, model, loss_fn, device):
    # 모델을 평가 모드로 설정
    model.eval()
    # 평가 과정에서 기울기를 계산하지 않도록 설정(메모리 사용을 줄이고 평가 속도를 높여준다.)
    with torch.no_grad():
        x, y = x_val_tensor.to(device), y_val_tensor.to(device)
        # 데이터 예측
        pred = model(x)
        # 예측 값 pred 와 실제 값 y 사이의 손실 계산
        eval_loss = loss_fn(pred, y).item()
    print(f"evaluete_err: {eval_loss:>7f}")
    return eval_loss


# CPU 또는 GPU 사용
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

# 데이터 준비
path = 'https://raw.githubusercontent.com/DA4BAM/dataset/master/boston.csv'
data = pd.read_csv(path)
data.head()

# 데이터 전처리
target = 'medv'
x = data.drop(target, axis=1)
y = data.loc[:, target]

# 데이터 분할
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2, random_state=20)

# 데이터 스케일링
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# 데이터 준비
train_loader, x_val_ts, y_val_ts = make_dataset(x_train, x_val, y_train, y_val, batch_size=4)

# 첫번째 배치만 로딩해서 살펴보기
for x, y in train_loader:
    print(f"Shape of x [rows, columns]: {x.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# 모델 생성
n_features = x.shape[1]

# 모델 구조 설계
model = nn.Sequential(
    nn.Linear(n_features, 3),  # hidden layer(input, output)
    nn.ReLU(),  # 활성화 함수 정의 -> ReLU
    nn.Linear(3, 1)  # 출력 Layer 설정
).to(device)

print(f"model 생성 : {model}")

# 오류 함수를 MSELoss 를 이용한다. MSE
loss_fn = nn.MSELoss()
# 모델의 가중치와 편향에 대해서 설정하는 함수
optimizer = Adam(model.parameters(), lr=0.01)

epochs = 20
# 학습 진행
for t in range(epochs):
    tr_loss = train(train_loader, model, loss_fn, optimizer, device)
    val_loss = evaluate(x_val_ts, y_val_ts, model, loss_fn, device)

    print(f"Epoch : {t + 1}, train loss : {tr_loss:.4f}, val loss : {val_loss:.4f}")

# 검증을 진행
result = evaluate(x_val_ts, y_val_ts, model, loss_fn, device)

print(f"result evaluate : {result}")
