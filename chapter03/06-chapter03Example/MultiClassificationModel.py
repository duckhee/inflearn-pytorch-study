"""
패션 아이템 이미지 10가지 분류하기
- 데이터 이미지(1 channel, 32x32) 6만장
- 10가지 클래스로 분류하기 위한 모델 생성
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import *

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import ToTensor

# numpy setting print option
np.set_printoptions(linewidth=np.inf)

# torch setting
torch.set_printoptions(linewidth=np.inf)


def make_dataset(x_train, x_val, y_train, y_val, batch_size=32):
    """
    :param x_train:
    :param x_val:
    :param y_train:
    :param y_val:
    :param batch_size:
    :return:
    """
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
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
    # 전체 데이터 셋 의 크기
    size = len(data_loader.dataset)
    # batch size 가져오기
    num_batches = len(data_loader)
    # 오차에 대한 저장할 변수
    tr_loss = 0
    # model 을 학습 모드로 변경
    model.train()

    # 학습 진행
    # batch는 현재 배치 번호를 의미
    # (X, y)는 입력 데이터와 라벨 정보를 의미
    for batch, (X, y) in enumerate(data_loader):
        # 어떤 것을 사용을 할지 정의 -> GPU, CPU를 이용할 지에 대해서 정의
        X, y = X.to(device), y.to(device)
        # 모델을 가지고 예측 및 학습
        pred = model(X)
        loss = loss_fn(pred, y)
        tr_loss += loss
        # 모델의 파라미터 변경 -> Back Propagation
        loss.backward()  # 역전파를 통해 모델의 각 파라미터에 대한 손실의 기울기를 계산
        optimizer.step()  # 옵티마이저가 계산된 기울기를 사용하여 모델의 파라미터를 업데이트
        optimizer.zero_grad()  # 옵티마이저의 기울기 값 초기화 -> 기울기가 누적이 되어서 잘못된 값 학습 방지

    # 평균 오차에 대해서 구하기
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
    plt.plot(epochs, tr_loss_list, label='Train', marker='.')
    plt.plot(epochs, val_loss_list, label='Validation', marker='.')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()


device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

print(f"using device: {device}")

train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=ToTensor(),
)

test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=ToTensor(),
)

classes = train_dataset.classes

batch_size = 64
train_data_loader = DataLoader(train_dataset, batch_size=batch_size)

for X, y in train_data_loader:
    print(f"Shape of X [batch, channels, height, width]: {X.shape}")
    print(f"Shape of y : {y.shape}, data type of y : {y.dtype}")
    break

x_val, x_test = test_dataset.data[:5000], test_dataset.data[5000:]
y_val, y_test = test_dataset.targets[:5000], test_dataset.targets[5000:]

x_val = x_val / 255
x_test = x_test / 255

x_val = x_val.view(5000, 1, 28, 28)
x_test = x_test.view(5000, 1, 28, 28)

features = 28 * 28

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(features, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001)

epochs = 20
tr_loss_list, val_loss_list = [], []

for t in range(epochs):
    tr_loss = train(train_data_loader, model, loss_fn, optimizer, device)
    val_loss, _ = evaluate(x_val, y_val, model, loss_fn, device)

    tr_loss_list.append(tr_loss)
    val_loss_list.append(val_loss)

    print(f"Epoch : {t + 1}, train loss: {tr_loss:4f}, validation loss: {val_loss:4f} ")

dl_learning_curve(tr_loss_list, val_loss_list)

_, pred = evaluate(x_test, y_test, model, loss_fn, device)

pred = nn.functional.softmax(pred, dim=1)
pred = np.argmax(pred.cpu().numpy(), axis=1)

cm = confusion_matrix(y_test.cpu().numpy(), pred)
print(f"confusion matrix : \r\n{cm}")

display_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
display_cm.plot()
plt.xticks(rotation=90)
plt.show()
