import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import *

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch import nn
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchsummary import summary


# 학습 하기 위한 데이터를 Tensor 형태로 만들어주는 함수
def make_data_set(x_train, x_val, y_train, y_val, batch_size=32):
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

    # 학습 시 이용을 하는 객체인 TensorDataset으로 만들어주기
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

    # 학습 시에 사용할 DataLoader 형태로 객체 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 학습 시 사용할 DataLoader와 검증을 위한 tensor 반환
    return train_loader, x_val_tensor, y_val_tensor


# 학습을 하는 함수
def train(data_loader, model, loss_fn, optimizer, device):
    """
    :param data_loader:
    :param model:
    :param loss_fn:
    :param optimizer:
    :param device:
    :return:
    """
    # 전체 데이터의 데한 개수
    size = len(data_loader)
    # batch 에 대한 크기 가져오기
    batch_size = len(data_loader.dataset)
    tr_loss = 0
    # 모델을 학습 모드로 변경
    model.train()

    # 배치에 대한 횟수만큼 학습 -> 학습 1회차
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        tr_loss += loss
        # 역전파를 이용한 오차에 대한 기울기 계산
        loss.backward()
        # 최적화 함수를 통한 업데이트
        optimizer.step()
        optimizer.zero_grad()
    tr_loss /= batch_size  # 오차 평균 구하기
    return tr_loss.item()


# 검증을 하기 위한 함수
def evaluate(x_val_tensor, y_val_tensor, model, loss_fn, device):
    """
    :param x_val_tensor:
    :param y_val_tensor:
    :param model:
    :param loss_fn:
    :param device:
    :return:
    """
    #  평가 모드로 변경
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
    plt.plot(epochs, val_acc_list, label='accuracy', marker='.')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()


# 디바이스 설정
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

print(f"using device : {device}")

# 학습 시 사용할 데이터 셋
train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=ToTensor()
)

# 픽셀 값에 대한 범위로 되어 있는 데이터를 0~1 사이 값으로 변환
train_data = train_dataset.data.numpy() / 255
test_data = test_dataset.data.numpy() / 255
# 라벨링에 대한 정보 변수
train_labels = train_dataset.targets.numpy()
test_labels = test_dataset.targets.numpy()

# 데이터 샘플링 -> 데이터에 대해서 학습 데이터, 검증 데이터, 테스트 데이터로 분리
x_train, _, y_train, _ = train_test_split(train_data, train_labels, test_size=40000, random_state=10,
                                          stratify=train_labels)
x_val, x_test, y_val, y_test = train_test_split(test_data, test_labels, test_size=5000, random_state=10,
                                                stratify=test_labels)

# 3차원 형태의 데이터를 4차원 데이터로 변형
x_train = x_train.reshape(20000, 1, 28, 28)
x_val = x_val.reshape(5000, 1, 28, 28)
x_test = x_test.reshape(5000, 1, 28, 28)

# tensor 형태로 데이터를 변환
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
x_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 학습 시 사용하기 위해서 data set 형태로 변환
train_tensorDS = TensorDataset(x_train, y_train)

# 현재 만들어진 데이터 셋에 대한 모양 확인
print(f"x train shape : {x_train.shape}, y train shape : {y_train.shape}")

# 분류할 정보들에 대한 출력
classes = train_dataset.classes
print(f"classes : {classes}")

# 한번 학습 시에 사용할 batch 에 대한 사이즈 설정
batch_size = 64

# 학습 시 사용하기 위한 DataLoader 생성
train_dataLoader = DataLoader(train_tensorDS, batch_size=batch_size, shuffle=True)

# DataLoader에 대한 데이터 모양 확인
for X, y in train_dataLoader:
    print(f"shape of X[batch, channels, height, width] :{X.shape}")
    print(f"shape of y : {y.shape}")
    break

# 입력 데이터의 크기
print(f"input data shape : {X.shape[1:]}, classes number : {len(classes)}")

# 분류할 종류의 갯수
n_clss = len(classes)
# Dropout에 대한 비율
dropout = 0.3
# 모델 선언
model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # size 28 / 2
    nn.Dropout2d(p=dropout),  # dropout 적용

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # size 14 / 2
    nn.Dropout2d(p=dropout),  # dropout 적용

    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # size 7 / 2
    nn.Dropout2d(p=dropout),  # dropout 적용

    nn.Flatten(),
    nn.Linear(in_features=128 * 3 * 3, out_features=128),
    nn.ReLU(),
    nn.Dropout(p=dropout),  # dropout 적용

    nn.Linear(in_features=128, out_features=64),
    nn.ReLU(),
    nn.Dropout(p=dropout),  # dropout 적용

    nn.Linear(in_features=64, out_features=n_clss)
).to(device)

# 손실 함수 정의
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# 모델에 대한 요약 정보 출력
model.to('cpu')  # -> CPU로 model을 가져오기
summary(model=model, input_size=X.shape[1:])
model.to(device)

# 학습 회수
epochs = 40
tr_loss_list, val_loss_list, val_acc_list = [], [], []

for t in range(epochs):
    tr_loss = train(data_loader=train_dataLoader, model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)
    val_loss, pred = evaluate(x_val_tensor=x_val, y_val_tensor=y_val, model=model, loss_fn=loss_fn, device=device)

    # accuracy 측정
    pred = nn.functional.softmax(pred, dim=1)
    pred = np.argmax(pred.cpu().numpy(), axis=1)
    acc = accuracy_score(y_true=y_val.numpy(), y_pred=pred)

    tr_loss_list.append(tr_loss)
    val_loss_list.append(val_loss)
    val_acc_list.append(acc)
    print(f"epoch {t + 1}, tr_loss : {tr_loss:.4f}, val_loss : {val_loss:.4f}, val_acc : {acc:.4f}")

# 학습 곡선 출력
dl_learning_curve(tr_loss_list, val_loss_list, val_acc_list)

# 예측 및 평가
_, pred = evaluate(x_test, y_test, model, loss_fn, device)
pred = nn.functional.softmax(pred, dim=1)
pred = np.argmax(pred.cpu().numpy(), axis=1)

# confusion matrix
cm = confusion_matrix(y_true=y_test.numpy(), y_pred=pred)
display_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
display_cm.plot()
plt.xticks(rotation=90)
plt.show()

# classification report
print('=' * 80)
print(f'Accuracy : {accuracy_score(y_test.numpy(), pred)}')
print('-' * 80)
print(classification_report(y_test.numpy(), pred, target_names=classes))
