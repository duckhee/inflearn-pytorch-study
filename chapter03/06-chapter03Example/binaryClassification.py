"""
회사 인사팀에서는 여러분들에게 직원의 이지 여부와 관련해서 분석을 요청하였다.
최근 이직율이 증가하는 것에 대해 우려를 갖고 있기에, 이직 여부에 영향을 주는 요인에 대해 분석하여, 이직할 것으로 보이는 직원들에 대해 회사를 떠나지 ㅇ낳도록 인사 프로그램을 준비하려고 한다.
어떤 직원이 이직할지에 대해서 예측하는 모델을 만들어보자.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import *

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam


def make_DataSet(x_train, x_val, y_train, y_val, batch_size=32):
    """
    :param x_train:
    :param x_val:
    :param y_train:
    :param y_val:
    :param batch_size:
    :return:
    """
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    # TensorDataset 생성 : 텐서 데이터셋으로 합치기
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, x_val_tensor, y_val_tensor


def train(data_loader, model, loss_fn, optimizer, device):
    """
    :param data_loader:
    :param model:
    :param loss_fn:
    :param optimizer:
    :param device:
    :return:
    """
    size = len(data_loader.dataset)
    batch_size = len(data_loader)
    tr_loss = 0
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        tr_loss += loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    tr_loss /= batch_size

    return tr_loss.item()


def evaluate(x_val_tensor, y_val_tensor, model, loss_fn, device):
    model.eval()

    with torch.no_grad():
        X, y = x_val_tensor.to(device), y_val_tensor.to(device)
        pred = model(X)
        eval_loss = loss_fn(pred, y).item()

    return eval_loss, pred


def dl_learning_curve(tr_loss_list, val_loss_list):
    """
    :param tr_loss_list:
    :param val_loss_list:
    :return:
    """
    epochs = list(range(1, len(tr_loss_list) + 1))
    plt.plot(epochs, tr_loss_list, label='Train', marker='.')
    plt.plot(epochs, val_loss_list, label='Validation', marker='.')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()


device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"using device: {device}")

path = "https://raw.githubusercontent.com/DA4BAM/dataset/master/Attrition_train_validation.CSV"
data = pd.read_csv(path)
# 이직할 것인지에 대한 여부 -> 가변수화
data['Attrition'] = np.where(data['Attrition'] == 'Yes', 1, 0)

print(f'sample data : \r\n{data.head(10)}')

target = 'Attrition'
x = data.drop(target, axis=1)

y = data.loc[:, target]
# 가변수화 진행
x = pd.get_dummies(x, columns=['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole',
                               'MaritalStatus', 'OverTime'],
                   drop_first=True)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2, random_state=20)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

train_loader, x_val_ts, y_val_ts = make_DataSet(x_train, x_val, y_train, y_val, 32)

for x, y in train_loader:
    print(f"shape of x [rows, columns] : {x.shape}")
    print(f"shape of y [rows, columns] : {y.shape}, y data type : {y.dtype}")
    break

n_features = x.shape[1]

model = nn.Sequential(
    nn.Linear(n_features, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
).to(device)

loss_fn = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.0001)

# 학습
epochs = 300

tr_loss_list, val_loss_list = [], []

for t in range(epochs):
    tr_loss = train(train_loader, model, loss_fn, optimizer, device)
    val_loss, _ = evaluate(x_val_ts, y_val_ts, model, loss_fn, device)

    tr_loss_list.append(tr_loss)
    val_loss_list.append(val_loss)
    print(f"epoch: {t + 1}, tr_loss: {tr_loss:4f}, val_loss: {val_loss:4f}")

dl_learning_curve(tr_loss_list, val_loss_list)

_, pred = evaluate(x_val_ts, y_val_ts, model, loss_fn, device)
pred = np.where(pred.cpu().numpy() > .5, 1, 0)

confusion_matrix = confusion_matrix(y_val_ts.numpy(), pred)

print(f"Accuracy : {accuracy_score(y_val_ts.cpu().numpy(), pred)}")
print('-' * 60)
print(classification_report(y_val_ts.cpu().numpy(), pred))
