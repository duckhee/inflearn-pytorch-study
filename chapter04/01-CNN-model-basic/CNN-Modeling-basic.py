import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import ToTensor
# 모델에 대한 요약을 출력하기 위한 라이브러리
from torchsummary import summary


# 학습용 데이터 생성 -> DataLoader, 검증 데이터 반환
def make_dataset(x_train, x_val, y_train, y_val, batch_size=32) -> []:
    """
    :param x_train:
    :param x_val:
    :param y_train:
    :param y_val:
    :param batch_size:
    :return:
    """
    x_train_tensor = torch.tensor(data=x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(data=y_train, dtype=torch.long)  # long = int64
    x_val_tensor = torch.tensor(data=x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(data=y_val, dtype=torch.long)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

    # DataLoader 생성
    data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return data_loader, x_val_tensor, y_val_tensor


# 데이터 학습
def train(data_loader, model, loss_fn, optimizer, device):
    """
    :param data_loader:
    :param model:
    :param loss_fn:
    :param optimizer:
    :param device:
    :return:
    """
    # 전체 데이터 수 가져오기
    size = len(data_loader.dataset)
    batch_size = len(data_loader)

    # 오차 반환 변수
    tr_loss = 0

    # model 학습 모드 설정
    model.train()

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        # 예측
        pred = model(X)
        # 오차 구하기
        loss = loss_fn(pred, y)
        tr_loss += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    tr_loss /= batch_size

    return tr_loss.item()


# 검증을 하는 함수
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
def dl_learning_curve(tr_loss_list, val_loss_list):
    """
    :param tr_loss_list:
    :param val_loss_list:
    :return:
    """
    epochs = list(range(1, len(tr_loss_list) + 1))
    plt.plot(epochs, tr_loss_list, 'b', label='Training loss', marker='.')
    plt.plot(epochs, val_loss_list, 'r', label='Validation loss', marker='.')

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid()
    plt.show()


# 디바이스 준비
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device : {device}")

# 사용할 데이터 다운로드
train_data_set = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=ToTensor(),
)

# 테스트 데이터 다운로드
test_data_set = datasets.MNIST(
    root="./data",
    download=True,
    train=False,
    transform=ToTensor(),
)

# 분류 모델의 종류 확인
print(f"model classes : {train_data_set.classes}")

# 학습에 사용이 되는 이미지 확인
n = 10
image, label = train_data_set.data[n], train_data_set.targets[n]

# 이미지 출력으로 확인 ->  pyplot 을 이용한 출력
plt.imshow(image, cmap='gray')
plt.title(f"label : {label}")
plt.show()

# datasets을 이용한 데이터 다운로드 시 ToTensor로 인해서 Tensor 데이터로 변환이 되어 있기 때문에 분할만 해주면 된다.
# DataLoader 형태로 만들어주면 된다.
train_dataloader = DataLoader(dataset=train_data_set, batch_size=64)

# 데이터 로더의 데이터 형태 확인을 위한 호출
for X, y in train_dataloader:
    print(f"Shape of X [batch, channels, height, width]: {X.shape}")
    print(f"Shape of y : {y.shape}, data type of y : {y.dtype}")
    break

# 데이터 검증 및 테스트 준비
"""
데이터 셋 분환
-> validation은 학습 시 epoch마다 성능 검증용으로 사용이 된다.
-> test는 모델 생성 후 최종 검증용으로 사용이 된다.
dataset의 data 속성으로 데이터를 가져오면, 원본 데이터가 나오게 된다.
-> scale이 적용이 안된 데이터이다.
-> 5000, 28, 28인 3차원 데이터 셋이다.
모델링에 사용하기 위해서는 두가지 전 처리가 필요하다.
-> 스케일링 원본 데이터가 0 ~ 255 까지 숫자이기 때문에 255를 나누어서 0~1 사이의 값으로 만들어준다.
-> 4차원 데이터로 변환을 해줘야 한다. 
=> 이미지가 가지고 있는 색상의 채널 값을 추가를 해줘야 한다.
"""
x_val, x_test = test_data_set.data[:5000], test_data_set.data[5000:]
y_val, y_test = test_data_set.targets[:5000], test_data_set.targets[5000:]

# test data의 모양 확인
print(f"validation shape of x : {x_val.shape}, validation shape of y : {y_val.shape} ")

# 데이터에 대한 스케일링
x_val = x_val / 255
x_test = x_test / 255

# 3차원 데이터를 4차원 데이터 형태로 변환
x_val = x_val.view(5000, 1, 28, 28)
x_test = x_test.view(5000, 1, 28, 28)

print(f"x_val shape : {x_val.shape}, x_test shape : {x_test.shape}")

# 분류를 할 종류의 갯수
n_classes = 10

# 모델 설계
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),  # input = 1 * 28 * 28, output = 8 * 28 * 28
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # input = 8 * 28 * 28, output = 8 * 14 * 14
    nn.Flatten(),  # 1차원 배열 형태로 변환
    nn.Linear(in_features=8 * 14 * 14, out_features=64),
    nn.ReLU(),
    nn.Linear(in_features=64, out_features=n_classes),
).to(device)
# summary 를 사용을 할 때에는 cuda 또는 cpu만 사용이 가능하다.
if device == 'cpu' or device == 'cuda':
    print(f"{summary(model, input_size=(1, 28, 28), device=device)}")
elif device == 'mps':
    print(f"{summary(model.cpu(), input_size=(1, 28, 28), device='cpu')}")
    model.to(device)
print("*" * 100)
# 오차 함수
loss_fn = nn.CrossEntropyLoss()
# 최적화 함수
optimizer = Adam(model.parameters(), lr=0.001)

# 전체 학습 수
epochs = 10
tr_loss_list, val_loss_list = [], []

for t in range(epochs):
    tr_loss = train(train_dataloader, model, loss_fn, optimizer, device)
    val_loss, _ = evaluate(x_val, y_val, model, loss_fn, device)

    tr_loss_list.append(tr_loss)
    val_loss_list.append(val_loss)

    print(f"epoch : {t + 1}, train loss : {tr_loss:4f}, validation loss : {val_loss:4f} ")

# 학습 곡선 확인
dl_learning_curve(tr_loss_list, val_loss_list)

# 모델 평가
_, pred = evaluate(x_test, y_test, model, loss_fn, device)

# 예측 결과 값 5개만 확인
print(f"predication 0 ~ 4 : {pred[:5]}")

# 테스트 결과 값 확률 값으로 변환
pred = nn.functional.softmax(pred, dim=1)
pred = np.argmax(pred.cpu().numpy(), axis=1)
print(f"probability : {pred}")

# confusion matrix
cm = confusion_matrix(y_test.cpu().numpy(), pred)
print(f"confusion matrix : \r\n{cm}")

# confusion matrix 시각화
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_data_set.classes)
disp.plot()
plt.xticks(rotation=90)
plt.show()

# 예측 결과 정보 출력 -> classification_report
print(f"accuracy score : \r\n{accuracy_score(y_test.cpu().numpy(), pred)}")
print('-' * 100)
print(f"classification report : \r\n{classification_report(y_test.cpu().numpy(), pred)}")

# print(f"model summary : \r\n{summary(model.to('cpu'), input_size=(1, 28, 28), device='cpu')}")
