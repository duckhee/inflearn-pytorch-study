import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import *

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchsummary import summary


# 학습을 위한 데이터 로더 생성
def make_data_set(x_train, x_validation, y_train, y_validation, batch_size=32):
    """
    :param x_train:
    :param x_validation:
    :param y_train:
    :param y_validation:
    :param batch_size:
    :return:
    """
    x_train_tensor = torch.tensor(x_train, dtype=torch.flaot32)
    x_validation_tensor = torch.tensor(x_validation, dtype=torch.flaot32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_validation_tensor = torch.tensor(y_validation, dtype=torch.long)

    train_data_set = TensorDataset(x_train_tensor, y_train)

    train_dataloader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)

    return train_dataloader, x_validation_tensor, y_validation_tensor


# 모델에 대한 학습을 하는 함수
def train(data_loader, model, loss_fn, optimizer, device):
    """
    :param data_loader:
    :param model:
    :param loss_fn:
    :param optimizer:
    :param device:
    :return:
    """
    size = len(data_loader)
    num_batches = len(data_loader.dataset)
    train_loss = 0
    model.train()

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss /= num_batches

    return train_loss.item()


# 모델에 대한 검증을 위한 함수
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


# 학습 곡선을 그려주는 함수
def dl_learning_curve(train_loss_list, validation_loss_list, validation_accuracy_list):
    epochs = list(range(1, len(train_loss_list) + 1))

    plt.plot(epochs, train_loss_list, label='Train', marker='.')
    plt.plot(epochs, validation_loss_list, label='Validation', marker='.')
    plt.plot(epochs, validation_accuracy_list, label='validation accuracy', marker='.')

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid()
    plt.show()


# device 설정
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

print(f"using device : {device}")

train_data_set = datasets.FashionMNIST(
    root='./data',
    download=True,
    train=True,
    transform=ToTensor()
)

test_data_set = datasets.FashionMNIST(
    root='./data',
    download=True,
    train=False,
    transform=ToTensor()
)

# 데이터 전처리 -> 픽셀 범위를 0과 1 사이로 변경 및 라벨링 작업
train_data = train_data_set.data.numpy() / 255
train_labels = train_data_set.targets.numpy()
test_data = test_data_set.data.numpy() / 255
test_labels = test_data_set.targets.numpy()

# 데이터 샘플링
x_train, _, y_train, _ = train_test_split(train_data, train_labels, test_size=40000, random_state=10,
                                          stratify=train_labels)
x_val, x_test, y_val, y_test = train_test_split(test_data, test_labels, test_size=5000, random_state=10,
                                                stratify=test_labels)

# 데이터 차원 변경
x_train = x_train.reshape(20000, 1, 28, 28)
x_val = x_val.reshape(5000, 1, 28, 28)
x_test = x_test.reshape(5000, 1, 28, 28)

# tesnsor로 데이터 변경
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
x_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# train data set으로 변환
train_tensorDataSet = TensorDataset(x_train, y_train)

# 데이터 셋의 모양 확인
print(f"x_train shape : {x_train.shape} size ({x_train.shape[2]}x{x_train.shape[3]}), y_train shape : {y_train.shape}")
print(f"x_val shape : {x_val.shape}, y_val shape : {y_val.shape}")

# 예측할 분류 종류 확인
classes = train_data_set.classes
print(f"class : {classes} length : {len(classes)}")

# train data loader 생성
batch_size = 64
train_dataloader = DataLoader(train_tensorDataSet, batch_size=batch_size)

# 배치 데이터 확인
for X, y in train_dataloader:
    print(f"shape of x [batch, channels, height, width] : {X.shape}")
    print(f"shape of y : {y.shape}, data type : {y.dtype}")
    break

# 레이어를 늘려서 모델링 실험

n_classes = len(classes)

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(in_features=8 * 7 * 7, out_features=n_classes),
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

summary(model.to('cpu'), input_size=(1, 28, 28))
model.to(device)

epochs = 20
tr_loss_list, val_loss_list, val_acc_list = [], [], []

for t in range(epochs):
    tr_loss = train(train_dataloader, model, loss_fn, optimizer, device)
    val_loss, pred = evaluate(x_val, y_val, model, loss_fn, device)

    pred = nn.functional.softmax(pred, dim=1)
    pred = np.argmax(pred.cpu().numpy(), axis=1)
    acc = accuracy_score(y_val.numpy(), pred)

    tr_loss_list.append(tr_loss)
    val_loss_list.append(val_loss)
    val_acc_list.append(acc)
    print(f"epoch : {t + 1}, train loss : {tr_loss:.4f}, validation loss : {val_loss:.4f}, accuracy : {acc:.4f}")

dl_learning_curve(tr_loss_list, val_loss_list, val_acc_list)

_, pred = evaluate(x_test, y_test, model, loss_fn, device)
pred = nn.functional.softmax(pred, dim=1)
pred = np.argmax(pred.cpu().numpy(), axis=1)

cm = confusion_matrix(y_test.numpy(), pred)
display_cm_matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
display_cm_matrix.plot()
plt.xticks(rotation=90)
plt.show()

print('=' * 80)
print(f"Accuracy : {accuracy_score(y_test.numpy(), pred):.4f}")
print('-' * 80)
print(classification_report(y_test.numpy(), pred, target_names=classes))

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(in_features=16 * 3 * 3, out_features=64),
    nn.ReLU(),
    nn.Linear(in_features=64, out_features=32),
    nn.ReLU(),
    nn.Linear(in_features=32, out_features=n_classes),
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

summary(model.to('cpu'), input_size=(1, 28, 28))

model.to(device)
epochs = 20
tr_loss_list, val_loss_list, val_acc_list = [], [], []

for t in range(epochs):
    tr_loss = train(train_dataloader, model, loss_fn, optimizer, device)
    val_loss, pred = evaluate(x_val, y_val, model, loss_fn, device)

    pred = nn.functional.softmax(pred, dim=1)
    pred = np.argmax(pred.cpu().numpy(), axis=1)
    acc = accuracy_score(y_val.numpy(), pred)

    tr_loss_list.append(tr_loss)
    val_loss_list.append(val_loss)
    val_acc_list.append(acc)
    print(f"epoch : {t + 1}, train loss : {tr_loss:.4f}, validation loss : {val_loss:.4f}, accuracy : {acc:.4f}")

dl_learning_curve(tr_loss_list, val_loss_list, val_acc_list)

_, pred = evaluate(x_test, y_test, model, loss_fn, device)
pred = nn.functional.softmax(pred, dim=1)
pred = np.argmax(pred.cpu().numpy(), axis=1)

cm = confusion_matrix(y_test.numpy(), pred)
display_cm_matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
display_cm_matrix.plot()
plt.xticks(rotation=90)
plt.show()
