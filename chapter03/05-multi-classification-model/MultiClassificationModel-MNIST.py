import numpy as np
import pandas as pd
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
    # data를 tensor 형태로 변경
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # int64
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # Tensor Dataset을 생성 -> tensor dataset 합치기
    train_tensor = TensorDataset(x_train_tensor, y_train_tensor)
    # Tensor Dataset을 불러올 수 있는 DataLoader 생성 -> shuffle은 데이터를 섞어서 보여주기 위한 옵션이다.
    train_loader = DataLoader(dataset=train_tensor, batch_size=batch_size, shuffle=True)
    # 데이터를 불러올 수 있는 Loader와 검증을 위한 tensor 데이터를 반환을 한다.
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


# 검증을 하는 함수
def evaluate(x_val_tensor, y_val_tensor, model, loss_fn, device):
    """
    :param x_va_tensor:
    :param y_val_tensor:
    :param model:
    :param loss_fn:
    :param device:
    :return:
    """
    # model 을 검증 모드로 변경
    model.eval()
    # 기울기를 계산하지 않도록 설정하기 위한 torch.no_grad() -> 메모리 사용을 줄이고 평가 속도를 높인다.
    with torch.no_grad():
        x, y = x_val_tensor.to(device), y_val_tensor.to(device)
        pred = model(x)
        eval_loss = loss_fn(pred, y).item()

    return eval_loss, pred


# 학습률을 확인하기 위한 그래프
def dl_learning_curve(tr_loss_list, val_loss_list):
    """
    :param tr_loss_list:
    :param val_loss_list:
    :return:
    """

    epochs = list(range(1, len(tr_loss_list) + 1))
    plt.plot(epochs, tr_loss_list, label='train_error', marker='.')
    plt.plot(epochs, val_loss_list, label='val_error', marker='.')

    plt.ylabel(ylabel='Loss')
    plt.xlabel(xlabel='Epochs')

    plt.legend()
    plt.grid()
    plt.show()


# 디바이스 준비 -> CPU OR GPU
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"using device : {device}")

# 데이터 가져오기 -> pytorch에서 제공을 하는 데이터 셋 (torch vision 에서 해당 데이터를 가져온다.)
# 해당 데이터들은 DataLoader를 통해서 가져오면 스케일된 데이터를 사용을 할 수 있고 스케일 되지 않은 데이터를 이용을 할 수 있다.
# Download training data from open datasets.
train_dataset = datasets.MNIST(
    root="data",
    train=True,  # 학습용 데이터를 다운로드 받기 위한 옵션
    download=True,
    transform=ToTensor(),  # 픽셀값을 [0,1] 사이로 정규화하고 텐서로 변환
)

# test 용 데이터는 DataLoader를 필요하지 않는다.
# Download test data from open datasets.
test_dataset = datasets.MNIST(
    root="data",
    train=False,  # test 데이터를 다운로드 받기 위한 옵션
    download=True,
    transform=ToTensor(),
)

# 데이터 확인을 위한 출력 -> data points는 데이터의 건수를 의미한다.
print(f"train_dataset : \r\n{train_dataset}, \r\n\r\ntest_data: {test_dataset}")

# dataset에 대한 모양을 확인하기 위한 출력
print(f"data shape : \r\n{train_dataset.data.shape}, target data shape : {train_dataset.targets.shape}")

# 라벨에 대한 정보는 classes로 정의가 되어 있다. -> 인덱스에 대한 label 정볼르 담고 있다.
print(f"label classes : {train_dataset.classes}")

# 데이터에 대한 확인을 위한 한 건 데이터 출력 -> 스케일된 이미지 데이터를 출력을 한다.
print(f"sample data : \r\n{train_dataset[0]}")

# dataset 에 저장이 되어 있는 이미지를 확인하기 위한 출력
n = 5000
image, label = train_dataset.data[n], train_dataset.targets[n]

plt.imshow(image, cmap='gray')
plt.title(label=f"Label : {label}")
plt.show()

# train 데이터를 dataloader 로 변환
batch_size = 64
# tensor 형태의 데이터를 사용을 하기 때문에 DataLoader로 변환을 해주기만 하면 된다
train_loader = DataLoader(train_dataset, batch_size=batch_size)

# 첫번쨰 데이터만 확인하기
for X, y in train_loader:
    # channel은 색상에 대한 정보를 의미한다. -> BGR 형태의 순선이다. 채널이 하나일 경우 1이 되고 흑백 이미지를 의미한다.
    print(f"Shape of X [batch, channels, height, width]: {X.shape}]")
    print(f"Shape of y : {y.shape}, data type of y : {y.dtype}")
    break

"""
data set 에 대한 분할을 한다.
- validation 은 학습 시, epoch마다 성능 검증용으로 사용이 된다.
- test는 모델 생성 후 최종 검증용으로 사용이 된다.
dataset의 data 속성으로 데이터를 가져오면 원본 데이터가 나오게 된다.
- 스케일링이 되지 않은 데이터가 나오게 된다.
- 5000, 28, 28 형태의 3차원 데이터 셋이 나오게 된다.
모데링에 사용하기 위해서 원본 데이터에 대한 두가지 전처리 과정이 필요하다.
- 스케일링을 통해서 원본 데이터가 0 ~ 255까지 숫자이기 때문에 255로 나누어서 확률 값으로 변환을 해준다.
- 4차원 배열 형태로 변경을 해줘야 한다. => 색상에 대한 정보를 포함해야 하기 때문이다.
"""
# 테스트 데이터에 대한 분할
x_val, x_test = test_dataset.data[:5000], test_dataset.data[5000:]
y_val, y_test = test_dataset.targets[:5000], test_dataset.targets[5000:]

# test 데이터에 대한 모양 확인
print(f" x val shape : {x_val.shape}, y val shape : {y_val.shape}")

# scaling 전처리
x_val = x_val / 255
x_test = x_test / 255

# 3차원 데이터를 4차원 데이터로 변환
x_val = x_val.view(5000, 1, 28, 28)  # [1*28*28 이미지 5000장] 형태로 구조를 변환을 한다.
x_test = x_test.view(5000, 1, 28, 28)

print(f"pre processing change dim x val shape : {x_val.shape}, x_test shape : {x_test.shape}")

# 모델 생성
# 학습에 사용할 파라미터가 이미지 한장에 있는 픽셀의 수이기 때문에 28 * 28 이다. (이미지의 크기가 28 * 28 이기 때문이다.)
n_features = 28 * 28
# 분류에 대한 라벨의 갯수
n_class = 10

# 모델 구조 설계
model = nn.Sequential(
    nn.Flatten(),  # 배열을 1차원 형태로 변환하기 위한 함수
    nn.Linear(n_features, n_class),  # 선형의 값을 가지고 있는 배열이기 때문이다.
).to(device)

# 모델의 구조에 대해서 출력
print(f"model is : {model}")

# 손실 함수와 최적화 함수 선언
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# 데이터 학습
epochs = 10
tr_loss_list, val_loss_list = [], []
# 학습
for t in range(epochs):
    tr_loss = train(train_loader, model, loss_fn, optimizer, device)
    val_loss, _ = evaluate(x_val, y_val, model, loss_fn, device)

    tr_loss_list.append(tr_loss)
    val_loss_list.append(val_loss)
    print(f"epoch: {t + 1}, tr_loss: {tr_loss:4f}, val_loss: {val_loss:4f}")

# 학습 곡선
dl_learning_curve(tr_loss_list, val_loss_list)

# 모델에 대한 평가
_, pred = evaluate(x_test, y_test, model, loss_fn, device)
pred[:5]

# 예측 결과를 각 클래스 별 확률 값으로 보기 위한 softmax 처리
pred = nn.functional.softmax(pred, dim=1)
# 가장 확률이 높은 클래스를 찾기 위해서 사용을 하는 함수가 np.argmax()이다.
pred = np.argmax(pred.cpu().numpy(), axis=1)
# 예측 결과에 대한 class 와 연결 시켜서 예측 결과 값으로 매칭 후 반환
print(f"predication value : {pred}")

cm = confusion_matrix(y_test.numpy(), pred)
print(f"confusion matrix : {cm}")

# confusion matrix 시각화
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
disp.plot()
plt.xticks(rotation=90)
plt.show()

print(classification_report(y_test.numpy(), pred))
