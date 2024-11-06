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
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

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
    # 모델 학습 모드 설정
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

# 하위 계층 비율, 교사 1명당 학생 비율, 범죄율을 가지고 집갑을 예측하는 모델을 만든다.
target = 'medv'
# 모델을 만들 때 사용이 되는 정보를 정의한다.
features = ['lstat', 'ptratio', 'crim']
# 모델에 대한 입력 값들을 담는다.
x = data.loc[:, features]
# 모델에 대한 정답을 가져온다.
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
train_loader, x_val_ts, y_val_ts = make_dataset(x_train, x_val, y_train, y_val, 32)

# 첫번째 배치만 로딩해서 살펴보기
# 배치의 사이즈가 32이므로 32개의 로우가 있다.
# 들어간 데이터의 정보가 3개이므로 3개의 컬럼이 있다.
for x, y in train_loader:
    print(f"Shape of x [rows, columns]: {x.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

n_feature = x.shape[1]

# 모델 구조 설계
model1 = nn.Sequential(
    nn.Linear(n_feature, 1),  # 입력이 되는 정보의 값과 출력 값에 대한 설정 -> 모델은 선형 모델이다.

).to(device)

print(model1)

# 오류를 정의하기 위한 함수 정의 -> 회귀 모델에 대한 오차 함수는 MSELoss()이다. (회귀 모델에 대한 오차 함수는 E의 값이 들어간다.)
loss_fn = nn.MSELoss()

# 최적화를 이용할 함수 정의 -> 학습률을 적용 시키기 위한 값은 lr이다.
optimizer = Adam(model1.parameters(), lr=0.1)

# 데이터에 대한 학습
# epochs 는 학습을 시킬 횟수에 대해서 정의한다.
# -> 학습된 데이터 반복 횟수를 정의한 것이다.
epochs = 50
# 오차율에 대한 값을 저장할 list
tr_loss_list, val_loss_list = [], []

# 학습을 진행
for t in range(epochs):
    # 데이터에 대한 학습
    tr_loss = train(dataloader=train_loader, model=model1, loss_fn=loss_fn, optimizer=optimizer, device=device)
    # 데이터에 대한 검증 진행
    val_loss, _ = evaluate(x_val_tensor=x_val_ts, y_val_tensor=y_val_ts, model=model1, loss_fn=loss_fn, device=device)

    # 오차율에 대한 리스트에 데이터 삽입
    tr_loss_list.append(tr_loss)
    val_loss_list.append(val_loss)
    # 학습에 대한 출력
    print(f"Epoch : {t + 1}, train loss : {tr_loss:4f}, val loss : {val_loss:4f}")

# 학습된 파라미터 확인
for name, param in model1.named_parameters():
    if param.requires_grad:
        # 여기서 weight는 각각의 정보에 곱해지는 가중치이고 bias는 더해주는 절편 값이다.
        print(f"Parameter : {name}, value : {param.data}")

# 학습에 대한 그래프 출력
dl_learning_curve(tr_loss_list, val_loss_list)

# 모델 평가
loss, pred = evaluate(x_val_tensor=x_val_ts, y_val_tensor=y_val_ts, model=model1, loss_fn=loss_fn, device=device)
# 오차 제곱의 평균 값이 MSE 이다.
print(f"MSE: {loss}")

# pred tensor 값을 numpy로 변경을 할 때 GPU를 이용하면 에러가 발생하므로 cpu로 변경해서 numpy로 변경하도록 해줘야 한다.
mae = mean_absolute_error(y_true=y_val_ts.numpy(), y_pred=pred.cpu().numpy())
mape = mean_absolute_percentage_error(y_true=y_val_ts.numpy(), y_pred=pred.cpu().numpy())

# MAE는 평균 오차를 의미한다. -> 단위 값을 곱하면 어느 정도의 오차 범위가 있다는 것을 보여줄 수 있다.
print(f"MAE : {mae}")
# MAPE 는 평균 오차율을 의미한다. -> 오차율이 몇 정도 된다고 이야기를 할 때 사용이 된다.
print(f"MAPE: {mape}")
