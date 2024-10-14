import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam  # 최적화 함수


# 토치 모델의 데이터 형태로 만들어주기 위한 함수
def make_dataset(x_train, x_val, y_train, y_val, batch_size=32):
    # 데이터를 텐서로 변경
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # TensorDataSet 생성 -> 텐서 데이터셋으로 합치기
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, x_val_tensor, y_val_tensor


# 학습을 진행하는 함수
def train(dataloader: DataLoader, model, loss_fn, optimizer, device):
    # 전체 데이터 셋의 크기
    size = len(dataloader.dataset)
    # 배치 크기 설정
    num_batches = len(dataloader)
    # 오차 저장 변수
    tr_loss = 0
    # 모델 학습
    model.train()
    # 배치 순서만큼 반복
    # batch : 현재 배치 번호, (X, y) : 입력 데이터와 레이블
    for batch, (X, y) in enumerate(dataloader):
        # 입력 데이터와 레이블을 지정된 장치(device, CPU 또는 GPU)로 연결
        X, y = X.to(device), y.to(device)

        # 예측
        pred = model(X)
        loss = loss_fn(pred, y)
        tr_loss += loss

        # 역전파를 통해 모델의 각 파라미터에 대한 손실의 기울기 계산
        loss.backward()
        # 최적화 함수가 계산한 기울기 값을 이용해서 모델의 파라미터 업데이트
        optimizer.step()
        # 옵티마이저의 기울기 값 초기화 기울기가 누적되는 것을 방지
        optimizer.zero_grad()

    # 모든 배치에서의 loss 에 대한 평균 구하기
    tr_loss /= num_batches
    return tr_loss.item()


# 검증을 하기 위한 함수
def evaluate(x_val_tensor, y_val_tensor, model, loss_fn, device):
    # 모델에 대한 평가 모드 설정
    model.eval()
    # 평가 과정에서 모델의 파라미터 값을 변경하지 않도록 설정
    with torch.no_grad():
        # 평가 과정에서 기울기를 계산하지 않도록 설정(메모리 사용을 줄이고 평가 속도를 높입니다.)
        x, y = x_val_tensor.to(device), y_val_tensor.to(device)
        pred = model(x)
        # 예측 값 pred 와 실제 값 y 사이의 차이를 계산
        eval_loss = loss_fn(pred, y).item()
    return eval_loss, pred


# 학습률에 대한 그래프 그리기 위한 함수
def dl_learning_curve(tr_loss_list, val_loss_list):
    epochs = list(range(1, len(tr_loss_list) + 1))
    plt.plot(epochs, tr_loss_list, label='train error', marker='.')
    plt.plot(epochs, val_loss_list, label='val error', marker='.')
    # Label 설정
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.show()


# cpu 혹은 gpu 사용
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
# 데이터 가져오기
path = 'https://raw.githubusercontent.com/DA4BAM/dataset/master/boston.csv'
data = pd.read_csv(path)
data.head()

target = 'medv'
features = ['lstat', 'ptratio', 'crim']
x = data.loc[:, features]
y = data.loc[:, target]

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2, random_state = 20)

# 스케일러 선언
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

train_loader, x_val_ts, y_val_ts = make_dataset(x_train, x_val, y_train, y_val, 32)

# 첫번째 배치만 로딩해서 살펴보기
for x, y in train_loader:
    print(f"Shape of x [rows, columns]: {x.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

n_feature = x.shape[1]

# 모델 구조 설계
model1 = nn.Sequential(

        ).to(device)

print(model1)
