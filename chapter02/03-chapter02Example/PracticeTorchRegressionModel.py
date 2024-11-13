from symtable import Function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from numpy import dtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam


# make train dataset function
def make_dataset(x_train, x_val, y_train, y_val, batch_size=32) -> []:
    """
    make train dataset Function
    """
    # data change tensor data
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    # label data change tensor
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    # tensor data add up
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    # 학습에서 사용할 DataLoader의 형태로 변경 -> suffle은 데이터를 섞어서 랜덤하게 선택을 하도록 하는 것
    train_dataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # return
    return train_dataLoader, x_val_tensor, y_val_tensor


# train function
def train(dataLoader: DataLoader, model: nn.Sequential, loss_fn, optimizer, device) -> float:
    """
    :param dataLoader: torch.utils.data.DataLoader
    :param model: torch.nn.Sequ
    :param loss_fn:
    :param optimizer:
    :param device:
    :return:
    """
    # get data size
    size = len(dataLoader.dataset)
    # get batch size
    num_batches = len(dataLoader)
    # train loss variable
    tr_loss = 0
    # model setting train mode
    model.train()
    # train one cycle
    for batch, (X, y) in enumerate(dataLoader):
        # setting device
        X, y = X.to(device), y.to(device)
        # predicate
        pred = model(X)
        # get train loss
        loss = loss_fn(pred, y)
        # train loss update
        tr_loss += loss
        # model update weight and bias
        # model에 대한 가중치 및 편차 변경
        # 역전파를 통해 모델의 각 파라미터에 대한 손실에 대한 기울기 구하기
        loss.backward()
        # opimizer udpate
        optimizer.step()
        # optimizer set inclination zero
        optimizer.zero_grad()
    # get loss avg
    tr_loss /= num_batches

    return tr_loss.item()


# evaluate function
def evaluate(x_val_tensor, y_val_tensor, model: nn.Sequential, loss_fn, device) -> []:
    """
    :param x_val_tensor:
    :param y_val_tensor:
    :param model:
    :param loss_fn:
    :param device:
    :return:
    """
    # model setting evaluation mode
    model.eval()
    # set not update weight and bias
    with torch.no_grad():
        # setting evaluation data device
        X, y = x_val_tensor.to(device), y_val_tensor.to(device)
        # predicated
        pred = model(X)
        # evaluation loss
        eval_loss = loss_fn(pred, y).item()

    return eval_loss, pred


# 평가에 대한 그래프로 보여주기 위한 함수
def dl_learning_curve(tr_loss_list: [], val_loss_list: []) -> None:
    # 학습률을 나타낼 변수
    epochs = list(range(1, len(tr_loss_list) + 1))

    plt.plot(epochs, tr_loss_list, label="train_error", marker='.')
    plt.plot(epochs, val_loss_list, label="val_error", marker='.')

    plt.ylabel(ylabel='Loss')
    plt.xlabel(xlabel='Epoch')
    plt.legend()
    plt.grid()
    plt.show()


# device setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Set device : {device}")

# car seat data path
path = 'https://raw.githubusercontent.com/DA4BAM/dataset/master/Carseats.csv'

# get data with pandas DataFrame
data = pd.read_csv(path)
# check data type print
print(f"data is header 5 line: \r\n{data.head(n=5)}")

# predication target is Sales
target = 'Sales'
# train data
x = data.drop(target, axis=1)
print(f"train data x = \r\n{x.head(n=5)}")
# label sales value
y = data.loc[:, target]
print(f"label data y = \r\n{y.head(n=5)}")

# 가변수화 -> 가변수화는 학습을 위해 문자 데이터를 숫자옇으로 변환을 하는 것을 말한다.
# 가변수할 데이터에 대한 column 명
cat_cols = ['ShelveLoc', 'Education', 'Urban', 'US']
x = pd.get_dummies(data=x, columns=cat_cols, drop_first=True)
print(f"가변수화 된 데이터 확인 :\r\n{x.head(n=5)}")

# data split train data, evaluation data
# test data 20%, random seed is random_state 20
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2, random_state=20)

# setting scalar
scaler = MinMaxScaler()
# data scale
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# data transform tensor
train_loader, x_val_ts, y_val_ts = make_dataset(x_train, x_val, y_train, y_val, batch_size=32)

# checking input layer parameter
for x, y in train_loader:
    print(f"Shape of x [rows, columns] : {x.shape}")
    print(f"Shape of y : {y.shape} {y.dtype}")
    break

# model setting
n_feature = x.shape[1]
print(f"model input parameter number : {n_feature}")

# make model
model = nn.Sequential(
    nn.Linear(n_feature, 8),  # input layer setting
    nn.ReLU(),
    nn.Linear(8, 4),  # hidden layer
    nn.ReLU(),
    nn.Linear(4, 1)  # output layer
).to(device)

# loss function setting
loss_fn = nn.MSELoss()
# optimizer function setting lr is learning rate
optimizer = Adam(model.parameters(), lr=0.001)

# train model
# 총 학습 횟수
epochs = 20
# loss list
tr_loss_list, val_loss_list = [], []

for t in range(epochs):
    tr_loss = train(train_loader, model, loss_fn, optimizer, device)
    val_loss, _ = evaluate(x_val_ts, y_val_ts, model, loss_fn, device)
    # add list
    tr_loss_list.append(tr_loss)
    val_loss_list.append(val_loss)

    print(f"Epoch : {t + 1}, train_loss : {tr_loss:4f}, val loss: {val_loss:4f}")

# show plt graph loss
dl_learning_curve(tr_loss_list, val_loss_list)
