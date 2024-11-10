import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import dtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam  # 최적화 함수 -> 활성화 함수


# dataset을 만들어주는 핫무 -> Tensor 형태로 데이터 변경
def make_dataset(x_train, x_val, y_train, y_val, batch_size=32) -> []:
    # tesnor로 변경
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    # 평가를 위한 데이터 tensor로 변경
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    # torch에서 데이터 학습을 할 때 사용을 하는 tesnor를 추가하는 DataLoader
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, x_val_tensor, y_val_tensor


# 학습을 하는 함수
def train(dataloader, model, loss_fn, optimizer, device) -> float:
    # 전체 데이터에 대한 set 의 크기
    size = len(dataloader.dataset)
    # 배치에 대한 크기
    num_batches = len(dataloader)
    # 오차율을 저장할 변수
    tr_loss = 0
    # model에 대한 모드 설정 -> 학습 모드로 설정
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


# 평가를 하는 함수
def evaluate(x_val_tensor, y_val_tensor, model, loss_fn, device) -> []:
    # 평가 모드로 변경
    model.eval()
    # 평가를 하는 과정에서 기울기를 계산하지 않도록 설정 -> 메모리 사용을 줄이고 평가 속도를 높인다.
    with torch.no_grad():
        # 평가 데이터와 레이블을 지정한 장치로 이동
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


# device 에 대한 CPU 또는 GPU 로 할지 가져오기
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
# 입력 파라미터 갯수
n_feature = x.shape[1]
# 모델 생성 -> hidden layer 1개
model = nn.Sequential(
    nn.Linear(n_feature, 2),
    nn.ReLU(),
    nn.Linear(2, 1),
).to(device)

# 모델에 대한 정보 출력
print(f"model : {model}")

# 오류를 정의하기 위한 함수 정의 -> 회귀 모델에 대한 오차 함수는 MSELoss()이다. (회귀 모델에 대한 오차 함수는 E의 값이 들어간다.)
loss_fn = nn.MSELoss()
# 최적화를 이용할 함수 정의 -> 학습률을 적용 시키기 위한 값은 lr이다.
optimizer = Adam(model.parameters(), lr=0.1)

# 학습을 진행할 횟수 정의
epochs = 100
# 오차에 대한 값을 담아줄 배열
tr_loss_list, val_loss_list = [], []
# 학습에 대한 회차 반복 횟수 진행
for t in range(epochs):
    # 모델 학습
    tr_loss = train(train_loader, model, loss_fn, optimizer, device)
    # 모델 검증
    val_loss, _ = evaluate(x_val_ts, y_val_ts, model, loss_fn, device)
    # 오차 값을 리스트에 담아주기
    tr_loss_list.append(tr_loss)
    val_loss_list.append(val_loss)
    # 학습에 대한 횟차 및 오차에 대한 값 출력
    print(f"Epoch {t + 1}, train loss : {tr_loss:4f}, val loss : {val_loss:4f}")

# 학습된 파라미터에 대한 값 확인
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Parameter: {name}, Value: {param.data}")

# 학습 곡선 확인을 위한 출력
dl_learning_curve(tr_loss_list, val_loss_list)

# 모델에 대한 평가
loss, pred = evaluate(x_val_ts, y_val_ts, model, loss_fn, device)

mae = mean_absolute_error(y_val_ts.numpy(), pred.cpu().numpy())
mape = mean_squared_error(y_val_ts.numpy(), pred.cpu().numpy())

print(f"MSE : {loss}")  # 오차의 값
print(f"MAE : {mae}")  # 평균 오차 값
print(f"MAPE : {mape}")  # 오차율 33%의 성능을 가진다.
