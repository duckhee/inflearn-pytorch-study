import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import ToTensor

import random as rd

from torchsummary import summary


def make_dataset(x_train, x_val, y_train, y_val, batch_size=32):
    """
    :param x_train:
    :param x_val:
    :param y_train:
    :param y_val:
    :param batch_size:
    :return:
    """
    x_train_tensor = torch.tensor(data=x_train, dtype=torch.float)
    x_val_tensor = torch.tensor(data=x_val, dtype=torch.float)

    y_train_tensor = torch.tensor(data=y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(data=y_val, dtype=torch.long)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_data_loader, x_val_tensor, y_val_tensor


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
    train_losses = 0

    model.train()

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        train_losses += loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_losses = train_losses / size

    return train_losses.item()


def evaluate(x_val_tensor, y_val_tensor, model, loss_fn, device):
    model.eval()

    with torch.no_grad():
        x, y = x_val_tensor.to(device), y_val_tensor.to(device)
        pred = model(x)
        eval_loss = loss_fn(pred, y)

    return eval_loss.item(), pred


def dl_learning_curve(train_loss, validation_loss):
    """
    :param train_loss:
    :param validation_loss:
    :return:
    """
    epochs = list(range(1, len(train_loss) + 1))
    plt.plot(epochs, train_loss, label='Train', marker='.')
    plt.plot(epochs, validation_loss, label='Validation', marker='.')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()


device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

print(f"using device : {device}")

train_data_set = datasets.MNIST(
    root='./data',
    train=True,
    transform=ToTensor(),
    download=True
)

test_data_set = datasets.MNIST(
    root='./data',
    train=False,
    transform=ToTensor(),
    download=True
)

# batch size setting
batch_size = 64
# train data-set setting DataLoader
train_loader = DataLoader(train_data_set, batch_size=batch_size)

# data split
x_val, x_test = test_data_set.data[:5000], test_data_set.data[5000:]
y_val, y_test = test_data_set.targets[:5000], test_data_set.targets[5000:]

# preprocessing
x_val = x_val / 255
x_test = x_test / 255

# make image data set change dimension 3D -> 4D
x_val = x_val.view(5000, 1, 28, 28)
x_test = x_test.view(5000, 1, 28, 28)

# model define
number_of_classification = 10

# kernel_size는 이미지 필터의 크기이다.
# nn.MaxPool2d를 진행을 하면 커널 사이즈만큼 입력 영상의 크기가 줄어든다.
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  # this is image filter
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # this is zipping feature
    nn.ReLU(),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(in_features=64 * 7 * 7, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=number_of_classification),
).to(device)

# summary 를 사용을 할 때에는 cuda 또는 cpu만 사용이 가능하다.
if device == 'cpu' or device == 'cuda':
    print(f"{summary(model, input_size=(1, 28, 28), device=device)}")
elif device == 'mps':
    print(f"{summary(model.cpu(), input_size=(1, 28, 28), device='cpu')}")
    model.to(device)
print("*" * 100)

# loss function define
loss_fn = nn.CrossEntropyLoss()
# optimizer function define
optimizer = Adam(model.parameters(), lr=0.001)

# 전체 학습 횟수 정의
epochs = 5
# 오차를 담아줄 리스트
tr_losses, val_losses = [], []

for i in range(epochs):
    tr_loss = train(train_loader, model, loss_fn, optimizer, device)
    val_loss, _ = evaluate(x_val, y_val, model, loss_fn, device)
    tr_losses.append(tr_loss)
    val_losses.append(val_loss)

    print(f"epoch : {i + 1}, train loss : {tr_loss:4f}, validation loss : {val_loss:4f}")

dl_learning_curve(tr_losses, val_losses)

_, pred = evaluate(x_test, y_test, model, loss_fn, device)

# 예측 값을 확인하기 위해서 softmax로 변환
pred = nn.functional.softmax(pred, dim=1)
# 평균 최대 값을 찾기 위해서 예측 값을 CPU로 가져와서 numpy 로 변환을 해준다.
pred = np.argmax(pred.cpu().numpy(), axis=1)

print(f"predication : {pred}")

# confusion metrix 출력
cm = confusion_matrix(y_test.cpu().numpy(), pred)

# confusion metrix display
display_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_data_set.classes)
display_cm.plot()
plt.xticks(rotation=90)
plt.show()

# classification report
print(f"accuracy : {accuracy_score(y_test.cpu().numpy(), pred)}")
print("-" * 100)
print(f"classification report : {classification_report(y_test.cpu().numpy(), pred)}")
