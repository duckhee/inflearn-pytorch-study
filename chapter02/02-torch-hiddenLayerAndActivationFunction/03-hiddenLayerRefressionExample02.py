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


def make_dataset(x_train, x_val, y_train, y_val, batch_size=32) -> []:
    # tensor 변환
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # 해당 모양에 대해서 설정하기 위한 view 함수 이용
    # 결과 값에 대한 tensor 변환
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    # tensor 형태로 데이터 합치기
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    # 학습에서 사용을 할 DataLoader 의 형태로 만들어주기
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # 생성한 DataLoader 반환
    return train_dataloader, x_val_tensor, y_val_tensor


# 학습을 위한 함수
def train(dataloader, model, loss_fn, optimizer, device) -> float:
    # 데이터에 대한 크기 가져오기
    size = len(dataloader.dataset)
    # 배치 사이즈 가져오기
    num_batches = len(dataloader)
    # 오차에 대해서 저장할 변수
    tr_loss = 0
    # model을 학습 모드로 설정
    model.train()
    # batch 크기만큼 반복을 하면서 학습 및 평가
    for batch, (X, y) in enumerate(dataloader):
        # 설정이 된 device 로 학습을 위한 준비
        X, y = X.to(device), y.to(device)
        # 모델로 예측
        pred = model(X)
        # 오차 함수를 이용한 정답과 오차를 확인
        loss = loss_fn(pred, y)
        # 오차에 대한 값을 더해서 구하기
        tr_loss += loss

        # model에 대한 가중치 및 편차 변경
        # 역전파를 통해 모델의 각 파라미터에 대한 손실에 대한 기울기 구하기
        loss.backward()
        # 옵티마지어가 계산된 기울기를 사용하여 모델의 파라미터 값 변경
        optimizer.step()
        # 다음 학습에 영향을 주지 않게 하기 위해서 최적화 함수 기울기 초기화
        optimizer.zero_grad()
    # 오차에 대해서 더한 값의 평균 구하기
    tr_loss /= num_batches

    return tr_loss.item()


# 평가하는 함수
def evaluate(x_val_tensor, y_val_tensor, model, loss_fn, device) -> []:
    # 평가 모드로 변경
    model.eval()
    # 모델의 값을 변경하지 않도록 설정
    with torch.no_grad():
        # 평가를 위한 데이터를 디바이스로 이동
        x, y = x_val_tensor.to(device), y_val_tensor.to(device)
        # 모델에 대한 예측 진행
        pred = model(x)
        # 평가 오차 값
        eval_loss = loss_fn(pred, y)
    return eval_loss, pred


# 평가에 대한 그래프로 보여주기 위한 함수
def dl_learning_curve(tr_loss_list, val_loss_list) -> None:
    # 학습률을 나타낼 변수
    epochs = list(range(1, len(tr_loss_list) + 1))

    plt.plot(epochs, tr_loss_list, label="train_error", marker='.')
    plt.plot(epochs, val_loss_list, label="val_error", marker='.')

    plt.ylabel(ylabel='Loss')
    plt.xlabel(xlabel='Epoch')
    plt.legend()
    plt.grid()
    plt.show()


# 학습에 사용할 디바이스 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

# 데이터 가져오기
path = 'https://raw.githubusercontent.com/DA4BAM/dataset/master/boston.csv'
data = pd.read_csv(path)
# 5줄의 데이터 가져오기
headerData = data.head()
print(f"{headerData}")

# 학습 데이터 및 검증 데이터로 나누기
# 필요한 데이터 준비하기
# 필요 없는 데이터 삭제
target = 'medv'
# 타겟에 대한 데이터의 열 삭제
x = data.drop(target, axis=1)
# 학습 시 확인할 값에 대한 분리 -> 정답에 대한 값
y = data.loc[:, target]
# 비율보다 데이터의 사이즈가 모델에 대한 정확도가 더 높아진다.
# 학습과 검증을 위한 비율을 8 : 2로 설정하기 위해서 test_size를 0.2로 정의 한다.
# 결과를 같게 나오기 위해서 사용이 되는 값이 random_state이다. -> 랜덤 시드 값이라고 이야기를 한다.
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2, random_state=20)

# 스케일러 선언
scaler = MinMaxScaler()
# 스케일링을 하고 학습용 데이터를 기준으로 피팅을 한 다음에 적용을 해서 결과를 담아 준다.
x_train = scaler.fit_transform(x_train)
# 적용만 해서 결과를 담아 달라는 것을 의미한다.
x_val = scaler.transform(x_val)

# 학습을 하기 위해서는 데이터 로더의 형태로 데이터를 만들어줘야 한다.
# -> 데이터 로더에 들어가는 데이터는 tensor의 형태를 가지는 데이터야 한다.
train_loader, x_val_ts, y_val_ts = make_dataset(x_train, x_val, y_train, y_val)

# 파라미터 값 가져오기
n_feature = x.shape[1]

# 모델 정의
model = nn.Sequential(
    nn.Linear(in_features=n_feature, out_features=10, bias=True),
    nn.ReLU(),
    nn.Linear(in_features=10, out_features=7, bias=True),
    nn.ReLU(),
    nn.Linear(in_features=7, out_features=5, bias=True),
    nn.ReLU(),
    nn.Linear(in_features=5, out_features=3, bias=True),
    nn.ReLU(),
    nn.Linear(in_features=3, out_features=1, bias=True),
).to(device)

# 모델에 대한 출력
print(f"model : {model}")

# 오차 함수 정의
loss_fn = nn.MSELoss()
# 옵티마이저 설정
optimizer = Adam(params=model.parameters(), lr=0.01)

# 학습을 위한 반복 횟수 정의
epochs = 100
# 오차를 받을 배열
tr_loss_list, val_loss_list = [], []

# 학습 시작
for t in range(epochs):
    # 학습
    tr_loss = train(train_loader, model, loss_fn, optimizer, device)
    # 평가
    val_loss, _ = evaluate(x_val_ts, y_val_ts, model, loss_fn, device)
    # 오차를 담아주기
    tr_loss_list.append(tr_loss)
    val_loss_list.append(val_loss)

    print(f"Epoch : {t + 1}, train loss: {tr_loss:4f}, val loss : {val_loss:4f}")

dl_learning_curve(tr_loss_list, val_loss_list)

# 모델 평가
loss, pred = evaluate(x_val_tensor=x_val_ts, y_val_tensor=y_val_ts, model=model, loss_fn=loss_fn, device=device)
# 오차 제곱의 평균 값이 MSE 이다.
print(f"MSE: {loss}")
mean_absolute_error(y_true=y_val_ts.numpy(), y_pred=pred.cpu().numpy())

# pred tensor 값을 numpy로 변경을 할 때 GPU를 이용하면 에러가 발생하므로 cpu로 변경해서 numpy로 변경하도록 해줘야 한다.
mae = mean_absolute_error(y_true=y_val_ts.numpy(), y_pred=pred.cpu().numpy())
mape = mean_absolute_percentage_error(y_true=y_val_ts.numpy(), y_pred=pred.cpu().numpy())

# MAE는 평균 오차를 의미한다. -> 단위 값을 곱하면 어느 정도의 오차 범위가 있다는 것을 보여줄 수 있다.
print(f"MAE : {mae}")
# MAPE 는 평균 오차율을 의미한다. -> 오차율이 몇 정도 된다고 이야기를 할 때 사용이 된다.
print(f"MAPE: {mape}")

