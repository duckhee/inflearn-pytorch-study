import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.display import display_latex
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import *
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torchvision import datasets
from torchvision.transforms import ToTensor

# numpy setting
np.set_printoptions(linewidth=np.inf)
# torch setting
torch.set_printoptions(linewidth=np.inf)


# 학습을 하기 위한 데이터를 만들어주는 함수
def train(x_train, x_val, y_train, y_val, batch_size=32):
    """
    :param x_train:
    :param x_val:
    :param y_train:
    :param y_val:
    :param batch_size:
    :return:
    """
    x_train_tensor = torch.tensor(data=x_train, dtype=torch.float32)
    x_val_tensor = torch.tensor(data=x_val, dtype=torch.long)
    y_train_tensor = torch.tensor(data=y_train, dtype=torch.float32)
    y_val_tensor = torch.tensor(data=y_val, dtype=torch.long)
    # TensorData set 생성 -> 텐서 데이터를 합치기
    train_tensor = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    # tensor 형태로 데이터를 가져올 수 있게 하기 위한 DataLoader
    train_loader = DataLoader(dataset=train_tensor, batch_size=batch_size, shuffle=True)

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
    # 전체 데이터 셋의 크기
    size = len(data_loader.dataset)
    # batch size 구하기
    batch_size = len(data_loader)
    # 오차 평균 구할 변수
    tr_loss = 0
    # 학습 모드로 변경
    model.train()
    # batch는 현재 배치 번호를 의미, X는 입력 데이터 y는 라벨 데이터
    for batch, (X, y) in enumerate(data_loader):
        # 학습에 이용할 장치 설정
        X, y = X.to(device), y.to(device)
        # 모델을 학습을 하기 위한 예측
        pred = model(X)
        # 오차 구하기
        loss = loss_fn(pred, y)
        tr_loss += loss

        # 파라미터 수정 -> 최적화 함수를 이용한 수정
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    tr_loss /= batch_size

    return tr_loss.item()


# 검증을 위한 함수
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


# 학습 곡선을 보여주기 위한 함수
def dl_learning_curve(tr_loss_list, val_loss_list):
    """
    :param tr_loss_list:
    :param val_loss_list:
    :return:
    """
    epochs = list(range(1, len(tr_loss_list) + 1))
    plt.plot(epochs, tr_loss_list, label='Training loss', marker='.')
    plt.plot(epochs, val_loss_list, label='Validation loss', marker='.')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()


# 디바이스 설정
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# 디바이스 확인을 위한 출력
print(f"using device : {device}")

# torch vision 에 있는 dataset 가져오기
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# 테스트 데이터 가져오기
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# train 데이터에 대한 DataLoader 생성 -> tensor 데이터이기 때문에 Loader에 넣어주면 된다.
batch_size = 64
train_dataLoader = DataLoader(train_dataset, batch_size=batch_size)

# dataloader에 대한 확인
for X, y in train_dataLoader:
    print(f"Shape of X [batch, channels, height, width]: {X.shape}")
    print(f"Shape of y : {y.shape}, y data type : {y.dtype}")
    break

# validation, test 준비
"""
데이터셋 분할 및 전처리 
"""
x_val, x_test = test_dataset.data[:5000], test_dataset.data[5000:]
y_val, y_test = test_dataset.targets[:5000], test_dataset.targets[5000:]

# 픽셀에 대해서 확률 값으로 변경
x_val = x_val / 255
x_test = x_test / 255

# 데이터를 차원 변경
x_val = x_val.view(5000, 1, 28, 28)
x_test = x_test.view(5000, 1, 28, 28)

n_features = 28 * 28
n_class = 10

# model 구조 설계
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(n_features, 100),
    nn.ReLU(),
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
).to(device)

# Loss Function과 Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# 학습을 할 횟수에 대해서 정의
epochs = 10
tr_loss_list, value_loss_list = [], []

for t in range(epochs):
    tr_loss = train(train_dataLoader, model, loss_fn, optimizer, device)
    val_loss, _ = evaluate(x_val, y_val, model, loss_fn, device)

    # 리스트에 추가
    tr_loss_list.append(tr_loss)
    value_loss_list.append(val_loss)
    print(f"Epoch : {t + 1}, train loss : {tr_loss:4f}, validation loss : {val_loss:4f}")

dl_learning_curve(tr_loss_list, value_loss_list)

# 모델에 대한 평가
_, pred = evaluate(x_test, y_test, model, loss_fn, device)

# 예측결과를 확률 값으로 변경
pred = nn.functional.softmax(pred, dim=1)
# 예측 결과를 CPU에 nupmy로 가져오기
pred = np.argmax(pred.cpu().numpy(), axis=1)

print(f"predicated result : {pred}")

# confusion matrix
cm = confusion_matrix(y_test.numpy(), pred)
print(f"confusion matrix : \r\n{cm}")

# confusion matrix 시각화
display_confusion_matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
display_confusion_matrix.plot()
plt.xticks(rotation=90)
plt.show()

# confusion report
print(f"confusion report : \r\n{classification_report(y_test.numpy(), pred)}")
