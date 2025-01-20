import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.optim import Adam
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchsummary import summary


def make_data_set(x_train, x_val, y_train, y_val, batch_size=32):
    """
    :param x_train:
    :param x_val:
    :param y_train:
    :param y_val:
    :param batch_size:
    :return:
    """
    x_train_tensor = torch.tensor(x_train, dtype=torch.floag32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_val_tensor = torch.tensor(x_val, dtype=torch.floag32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # tensor data set
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    # make data loader
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

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    tr_loss /= batch_size
    return tr_loss.item()


def evaluate(x_val_tensor, y_val_tensor, model, loss_fn, device):
    """
    :param x_val_tensor:
    :param y_val_tensor:
    :param model:
    :param device:
    :return:
    """

    model.eval()

    with torch.no_grad():
        x, y = x_val_tensor.to(device), y_val_tensor.to(device)
        pred = model(x)
        eval_loss = loss_fn(pred, y).item()

    return eval_loss, pred


def dl_learning_curve(tr_loss_list, val_loss_list, val_acc_list):
    """
    :param tr_loss_list:
    :param val_loss_list:
    :param val_acc_list:
    :return:
    """

    epochs = list(range(1, len(tr_loss_list) + 1))
    plt.plot(epochs, tr_loss_list, label='Train', marker='.')
    plt.plot(epochs, val_loss_list, label='Validation', marker='.')
    plt.plot(epochs, val_acc_list, label='Validation', marker='.')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()


device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"using device : {device}")

train_data_set = datasets.FashionMNIST(
    root="./data",
    download=True,
    train=True,
    transform=ToTensor()
)

test_data_set = datasets.FashionMNIST(
    root="./data",
    download=True,
    train=False,
    transform=ToTensor()
)

# 데이터와 레이블 추출 -> 255로 나누는 이유는 해당 값을 픽셀을 0 ~ 1 범위로 변경을 해주기 위한 전처리
train_data = train_data_set.data.numpy() / 255
train_labels = train_data_set.targets.numpy()
test_data = test_data_set.data.numpy() / 255
test_labels = test_data_set.targets.numpy()

# 데이터에 대한 샘플링, 층화 추출
x_train, _, y_train, _ = train_test_split(train_data, train_labels, test_size=40000, random_state=10,
                                          stratify=train_labels)
x_val, x_test, y_val, y_test = train_test_split(test_data, test_labels, test_size=5000, random_state=10,
                                                stratify=test_labels)

# 3차원 데이터를 4차원 데이터로 변환 -> 이미지 개수 채널 수 이미지 가로 이미지 세로
x_train = x_train.reshape(20000, 1, 28, 28)
x_val = x_val.reshape(5000, 1, 28, 28)
x_test = x_test.reshape(5000, 1, 28, 28)

# tensor 형태로 변환
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
x_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# train dataset 으로 변환
train_TensorDS = TensorDataset(x_train, y_train)

print(f"x train shape : {x_train.shape}, y train shape : {y_train.shape}")
print(f"x val shape : {x_val.shape}, y val shape : {y_val.shape}")

# 분류할 종류 개수 가져오기
classes = train_data_set.classes
print(f"classes : {classes}")

batch_size = 64

train_dataloader = DataLoader(train_TensorDS, batch_size=batch_size)
input_shape = ()
# 데이터 구조를 확인하기 위한 출력
for X, y in train_dataloader:
    print(f"Shape of X [batch, channels, height, width]: {X.shape} ")
    print(f"Shape of y: {y.shape} {y.dtype}")
    input_shape = X.shape[1:4]
    break

n_classes = len(classes)

# 모델 선언 -> 현재 모델은 과적합 모델이다.
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # size 14 x 14
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # size 7 x 7
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # size 3 x 3
    nn.Flatten(),
    nn.Linear(in_features=128 * 3 * 3, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=64),
    nn.ReLU(),
    nn.Linear(in_features=64, out_features=n_classes)
).to(device)

# loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

summary(model=model.to("cpu"), input_size=input_shape)
model.to(device)

# 과적합을 시키기 위한 횟수 정의
epoch = 40
tr_loss_list, val_loss_list, val_acc_list = [], [], []

# early stopping 을 하기 위한 변수
patience = 3  # 기다릴 횟수를 지정한 값
best_error = float('inf')  # 초기 값은 무한대로 설정
counter = 0

for t in range(epoch):
    tr_loss = train(train_dataloader, model, loss_fn, optimizer, device)
    val_loss, pred = evaluate(x_val, y_val, model, loss_fn, device)
    # accuracy 측정
    pred = nn.functional.softmax(pred, dim=1)
    pred = np.argmax(pred.cpu().numpy(), axis=1)
    acc = accuracy_score(y_true=y_val.numpy(), y_pred=pred)

    # 리스트에 추가
    tr_loss_list.append(tr_loss)
    val_loss_list.append(val_loss)
    val_acc_list.append(acc)

    print(
        f"epoch : {t + 1}, train loss : {tr_loss:.4f}, validation loss : {val_loss:.4f}, validation accuracy : {acc:.4f}")
    # early stopping 이 필요한지 확인하기 위한 로직
    if val_loss > best_error:
        counter += 1
        print(f"----> early stopping status, best_loss : {best_error:.4f}, counter : {counter}")
    else:
        best_error = val_loss
        counter = 0

    # 조기 종료 조건 확인
    if counter >= patience:
        print("Early Stopping")
        # best model 에 대한 저장
        torch.save(obj=model, f="./fashion_mnist_model.pt")
        break

# 학습 곡선 확인을 위한 출력
dl_learning_curve(tr_loss_list, val_loss_list, val_acc_list)

# 모델에 대한 평가
_, pred = evaluate(x_test, y_test, model, loss_fn, device)

pred = nn.functional.softmax(pred, dim=1)
pred = np.argmax(pred.cpu().numpy(), axis=1)

# confusion metrix 구현
cm = confusion_matrix(y_test.numpy(), pred)
print(f"confusion matrix : {cm}")

# confusion metrix 시각화
display_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
display_cm.plot()
plt.xticks(rotation=90)
plt.show()

print(f"accuracy : {accuracy_score(y_test.numpy(), pred)}")
print("-" * 100)
print(classification_report(y_test.numpy(), pred, target_names=classes))

# 저장된 모델 사용을 하기
load_model = torch.load(f="./fashion_mnist_model.pt", weights_only=False)

# 불러온 모델로 예측
_, pred = evaluate(x_test, y_test, load_model, loss_fn, device)
pred = nn.functional.softmax(pred, dim=1)
pred = np.argmax(pred.cpu().numpy(), axis=1)

# confusion metrix
load_cm = confusion_matrix(y_test.numpy(), pred)
display_load_cm = ConfusionMatrixDisplay(confusion_matrix=load_cm, display_labels=classes)
display_load_cm.plot()
plt.xticks(rotation=90)
plt.show()

print("=" * 80)
print(f"Accuracy : {accuracy_score(y_test.numpy(), pred)}")
print("=" * 80)
print(f"Classification Report : {classification_report(y_test.numpy(), pred)}")
