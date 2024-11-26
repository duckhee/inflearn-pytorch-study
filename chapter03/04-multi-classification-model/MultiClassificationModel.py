from numbers import Number

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam


def make_data_set(x_train, x_val, y_train, y_val, batch_size=32):
    """
    :param x_train:
    :param x_val:
    :param y_train:
    :param y_val:
    :param batch_size:
    :return:
    """
    # train data change tensor data
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, x_val_tensor, y_val_tensor


def train(dataloader, model, loss_fn, optimizer, device):
    """
    :param dataloader:
    :param model:
    :param loss_fn:
    :param optimizer:
    :param device:
    :return:
    """
    # data set size
    size = len(dataloader.dataset)
    # bacth size
    num_batches = len(dataloader)
    tr_loss = 0
    # model train mode setting
    model.train()
    # data load
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # predicated
        pred = model(X)
        loss = loss_fn(pred, y)
        tr_loss += loss
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    tr_loss /= num_batches

    return tr_loss.item()


def evaluate(x_val_tensor, y_val_tensor, model, loss_fn, device):
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

    with torch.no_grad():
        x, y = x_val_tensor.to(device), y_val_tensor.to(device)
        pred = model(x)
        eval_loss = loss_fn(pred, y).item()
    return eval_loss, pred


def dl_learning_curve(tr_loss_list, val_loss_list):
    """
    :param tr_loss_list:
    :param val_loss_list:
    :return:
    """
    epochs = list(range(1, len(tr_loss_list) + 1))
    plt.plot(epochs, tr_loss_list, label='train_err', marker='.')
    plt.plot(epochs, val_loss_list, label='val_err', marker='.')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.show()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 꽃받침의 길이와 폭, 꽃잎의 길이와 폭을 가지고 종류를 맞추기 위한 데이터이다.
path = "https://raw.githubusercontent.com/DA4BAM/dataset/master/iris.csv"
data = pd.read_csv(path)
print(f"IRIS Data : \n{data.head()}")

# data setting
target = 'Species'
x = data.drop(target, axis=1)
y = data.loc[:, target]

# 다중 분류를 하기 위해서는 문자열에 대한 값을 숫자 형태로 변경을 해줘야 한다.
# Integer Encoding이라고 부르기도 한다. Sklearn에 있는 LabelEncoder를 이용을 한다.
le = LabelEncoder()
y = le.fit_transform(y)

print(f"Label Encoder :\n {le.classes_}, Label :\n {y[:1]}")
print(f"Inverse Label Encoder :\n {le.inverse_transform(y)[:1]}")

# split test data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.3, random_state=20)

# scale
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

# make data set
train_loader, x_val_ts, y_val_ts = make_data_set(x_train, x_val, y_train, y_val, batch_size=32)

for x, y in train_loader:
    print(f"Shape of x[rows, columns] : {x.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

n_feature = x.shape[1]
# output node number get
n_class = len(le.classes_)

model = nn.Sequential(
    nn.Linear(n_feature, n_class)
).to(device)

print(f"model is : \n{model}")

# multi classification loss function cross entropy
loss_fn = nn.CrossEntropyLoss()
# optimizer setting Adam
optimizer = Adam(model.parameters(), lr=0.1)

epochs = 100

tr_loss_list, val_loss_list = [], []

for t in range(epochs):
    tr_loss = train(train_loader, model, loss_fn, optimizer, device)
    val_loss, _ = evaluate(x_val_ts, y_val_ts, model, loss_fn, device)
    tr_loss_list.append(tr_loss)
    val_loss_list.append(val_loss)

    print(f"Epoch {t + 1} , train_loss : {tr_loss:4f}, val_loss : {val_loss:4f}")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Parameter : {name}, Value : {param.data}")

dl_learning_curve(tr_loss_list, val_loss_list)

# predicated model
_, pred = evaluate(x_val_ts, y_val_ts, model, loss_fn, device)

# multi classification pred 2D
print(f"Predication 2D : {pred.numpy()[:5]}")

# change softmax dim=1 is row, dim=0 is column
pred = nn.functional.softmax(pred, dim=1)
print(f"Predication Value : {pred[:5]}")

# most bigist index
pred = np.argmax(pred, axis=1)
print(f"most big index : {pred[:5]}")

get_confusion = confusion_matrix(y_val_ts.numpy(), pred)

print(f"confusion matrix : {get_confusion}")

# classification_report에 target에 대한 이름을 지정을 하면 해당 이름으로 출력을 해준다.
print(f"classification report : {classification_report(y_val_ts.numpy(), pred, target_names=le.classes_)}")
