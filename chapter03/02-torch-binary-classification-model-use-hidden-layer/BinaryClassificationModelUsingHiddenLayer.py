import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset


# make data set -> tensor
def make_dataset(x_train, x_val, y_train, y_val, batch_size=32):
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    # view(-1, 1)의 기능은 numpy의 reshape와 동일한 기능을 한다. 2차원 데이터 형태로 만들기 위한 설정이다.
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    # mini batch size로 학습을 하기 위해서 DataLoader를 만들기 위한 데이터 생성
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 문제를 잘라서 줄 필요가 없기 때문이다.
    # validation에 대한 반환을 mini batch 형식인 DataLoader로 반환을 하지 않는 이유는 한꺼번에 계산해서 전체에 대한 예측 값에 대한 오차를 계산을 하면 되기 때문이다.
    return train_loader, x_val_tensor, y_val_tensor


# 학습을 위한 함수
def train_model(data_loader: DataLoader, model: nn.Sequential, loss_fn, optimizer: torch.optim, device):
    """
    :param data_loader:
    :param model:
    :param loss_fn:
    :param optimizer:
    :param device:
    :return:
    """
    # 전체 데이터 셋에 대한 크기
    size = len(data_loader.dataset)
    # batch size
    num_batches = len(data_loader)
    # loss
    tr_loss = 0
    # model setting train mode
    model.train()
    # mini batch start
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        # data forward
        # Compute predicated error
        pred = model(X)
        loss = loss_fn(pred, y)
        tr_loss += loss

        # update weight and bias -> 역전파를 이용한 모델의 각 파라미터에 대한 손실의 기울기 계산
        # Back propagation
        loss.backward()
        # update weight and bias
        optimizer.step()
        optimizer.zero_grad()
    tr_loss /= num_batches

    return tr_loss.item()


# 검증을 위한 함수
def evaluate_model(x_val_tensor, y_val_tensor, model, loss_fn, device):
    """
    :param x_val_tensor:
    :param y_val_tensor:
    :param model:
    :param loss_fn:
    :param device:
    :return:
    """
    # model setting evaluate mode
    model.eval()
    # not update weight
    with torch.no_grad():
        x, y = x_val_tensor.to(device), y_val_tensor.to(device)
        pred = model(x)
        eval_loss = loss_fn(pred, y).item()

    return eval_loss, pred


def learning_curve(tr_loss_list, val_loss_list) -> None:
    """
    :param tr_loss_list:
    :param val_loss_list:
    :return:
    """
    epochs = list(range(1, len(tr_loss_list) + 1))
    plt.plot(epochs, tr_loss_list, 'b', label='Training loss')
    plt.plot(epochs, val_loss_list, 'r', label='Validation loss')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.show()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using {device} device")

# get data -> 타이타닉에 대한 생존 데이터를 가지고 모델링 생성
path = "https://raw.githubusercontent.com/DA4BAM/dataset/master/titanic.3.csv"
data = pd.read_csv(path)
print(f"get data head 5 : \r\n {data.head()}")

# 데이터에서 예측할 값과 모델에 대한 파라미터로 나누기
target = 'Survived'
features = ['Sex', 'Age', 'Fare']
x = data.loc[:, features]
y = data.loc[:, target]

# 범주형 데이터에 대해서 가변수화
x = pd.get_dummies(x, columns=['Sex'], drop_first=True)
print(f"Dummy variables : \r\n{x.head()}")

# 학습 데이터 분할 -> test_size는 테스트 데이터와 검증 데이터를 비율로 분할 random_state는 랜덤에 대한 시드 값이 된다.
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.3, random_state=20)

# 데이터에 대한 스케일링
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# 모델링을 위한 데이터 tensor 변환
train_loader, x_val_ts, y_val_ts = make_dataset(x_train, x_val, y_train, y_val, batch_size=32)

# 첫번째 배치로 로딩 확인
for x, y in train_loader:
    print(f"Shape of x [rows, columns] : {x.shape}")
    print(f"Shape of y :{y.shape} {y.dtype}")
    break

# model parameter 값에 대한 정의
x_feature = x.shape[1]

model = nn.Sequential(
    nn.Linear(x_feature, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
    nn.Sigmoid()
).to(device)

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 50

tr_loss_list, val_loss_list = [], []
# 학습
for t in range(epochs):
    tr_loss = train_model(train_loader, model, loss_fn, optimizer, device)
    val_loss, _ = evaluate_model(x_val_ts, y_val_ts, model, loss_fn, device)

    tr_loss_list.append(tr_loss)
    val_loss_list.append(val_loss)

    print(f"Epoch {t + 1} train loss : {tr_loss:4f}, val loss: {val_loss:4f}")

# 학습된 데이터 확인
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Parameter : {name}, value : {param.data}")

learning_curve(tr_loss_list, val_loss_list)

# 모델에 대한 평가
_, pred = evaluate_model(x_val_ts, y_val_ts, model, loss_fn, device)

print(f"predication : {pred.numpy()[:5]}")
# 이진 분류 형태이기 때문에 예측 값에 대한 변환
pred = np.where(pred > 0.5, 1, 0)
print(f"transform predication : {pred[:5]}")

# 해당 값을 가지고 이진 분류에 대한 값 확인
cf_matrix = confusion_matrix(y_val_ts, pred)
# 에측 값에 대한 실제 값 과 예측 값으로 이루어진 행렬이다.
print(f"confusion matrix : \r\n{cf_matrix}\r\n")
# f1-score 는 (맞춘 비율->confusion_matrix에서 대각 성분)/전체 데이터
# recall의 경우 실제 데이터에서 어느 정도 모델이 맞췄는지에 대해서 나타낸다.
# precision의 경우 모델이 예측을 했는데 진짜로 예측한대로 된 데이터에 대한 비율을 말한다.
# f1-score의 경우 precision과 recall에 대한 조화 평균을 의미한다.
print(classification_report(y_val_ts.numpy(), pred))
