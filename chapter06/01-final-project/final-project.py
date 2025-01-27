import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.config_init import max_colwidth_doc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import *

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torchvision.transforms import ToTensor
from torchvision import datasets
from torchsummary import summary

import CustomModel


def train(dataloader, model, loss_fn, optimizer, device):
    """
    :param dataloader:
    :param model:
    :param loss_fn:
    :param optimizer:
    :param device:
    :return:
    """

    size = len(dataloader.dataset)
    batch_size = len(dataloader)

    tr_loss = 0

    model.train()

    for batch, (X, y) in enumerate(dataloader):
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


def dl_learning_curve(tr_loss_list, eval_loss_list, val_acc_list):
    """
    :param tr_loss_list:
    :param eval_loss_list:
    :param val_acc_list:
    :return:
    """
    epochs = list(range(1, len(tr_loss_list) + 1))

    plt.plot(epochs, tr_loss_list, label='Training loss')
    plt.plot(epochs, eval_loss_list, label='Validation loss')
    plt.plot(epochs, val_acc_list, label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()


device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"using device : {device}")

train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=ToTensor()
)

# data set 확인
print(f"f train data set : \n {train_dataset}")
print(f"f test data set : \n {test_dataset}")

# train data set 에 대한 모양 확인
print(f"train data set shape : {train_dataset.data.shape}")
print(f"train dataset target list : {train_dataset.targets[:10]}")

# 분류할 갯수 종류 확인
print(f"classes information : {train_dataset.classes}")

# 분류할 모델의 개수
classes_number = len(train_dataset.classes)

# 트래인 데이터에 대한 이미지 확인

sample_number = 29

sample_image, sample_label = train_dataset.data[sample_number], train_dataset.targets[sample_number]

plt.imshow(sample_image)
plt.title(f"{sample_label}")
plt.show()

# 학습을 위한 데이터 준비

# batch size 정의
batch_size = 64

# data loader 로 만들기 학습을 위한 데이터
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# data loader 의 데이터 확인
for temp_x, temp_y in train_data_loader:
    print(f"shape of X [N, C, H, W] : {temp_x.shape}")
    print(f"shape of y : {temp_y.shape}, y data type : {temp_y.dtype}")
    break

# 검증용 데이터와 테스트 데이터 분할
test_data = test_dataset.data
test_target = np.array(test_dataset.targets)

# 데이터 위치 조절 -> 차원에 대한 위치 변경
test_data = test_data.transpose((0, 3, 1, 2))

test_data = torch.tensor(test_data, dtype=torch.float32)
test_target = torch.tensor(test_target, dtype=torch.int64)

x_val, x_test = test_data[:5000], test_data[5000:]
y_val, y_test = test_target[:5000], test_target[5000:]

# 검즈용 데이터와 테스트 용 데이터 확인
print(f"val shape : {x_val.shape}, {y_val.shape}")
print(f"test shape : {x_test.shape}, {y_test.shape}")

# 검즈용 데이터 크기 스케일링
x_val = x_val / 255.0
x_test = x_test / 255.0

# 선언적 모델 선언

model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(in_features=64 * 8 * 8, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=classes_number),
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

model.to('cpu')
summary(model, input_size=(3, 32, 32))
model.to(device)

# 데이터 학습
# 전체 데이터를 학습하기 위한 횟수
epochs = 20

tr_loss_list, val_loss_list, val_acc_list = [], [], []

for t in range(epochs):
    tr_loss = train(train_data_loader, model, loss_fn, optimizer, device)
    eval_loss, pred = evaluate(x_val, y_val, model, loss_fn, device)

    # 예측 값 측정
    pred = nn.functional.softmax(pred, dim=1)
    pred = np.argmax(pred.cpu().numpy(), axis=1)
    acc = accuracy_score(y_val.numpy(), pred)

    tr_loss_list.append(tr_loss)
    val_loss_list.append(eval_loss)
    val_acc_list.append(acc)

    print(f"Epoch : {t + 1}, train loss : {tr_loss:.4f}, val loss : {eval_loss:.4f}, val accuracy : {acc:.4f}")

dl_learning_curve(tr_loss_list, val_loss_list, val_acc_list)

# 예측
_, pred = evaluate(x_test, y_test, model, loss_fn, device)
pred = nn.functional.softmax(pred, dim=1)
pred = np.argmax(pred.cpu().numpy(), axis=1)

# confusion matrix
cm = confusion_matrix(y_test.numpy(), pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
disp.plot()
plt.xticks(rotation=90)
plt.show()

# classification report
print('=' * 80)
print(f'Accuracy : {accuracy_score(y_test.numpy(), pred)}')
print('-' * 80)
print(classification_report(y_test.numpy(), pred, target_names=train_dataset.classes))

custom = CustomModel.CustomModel(3, classes_number, 0.3)

loss_fn_custom_model = nn.CrossEntropyLoss()
optimizer_custom_model = Adam(custom.parameters(), lr=0.001)

custom.to('cpu')
summary(custom, input_size=(3, 32, 32))
custom.to(device)

# 데이터 학습
# 전체 데이터를 학습하기 위한 횟수
epochs = 100
# early stop 을 위한 변수
patience = 3
best_error = float('inf')
counter = 0

tr_loss_list, val_loss_list, val_acc_list = [], [], []

for t in range(epochs):
    tr_loss = train(train_data_loader, custom, loss_fn_custom_model, optimizer_custom_model, device)
    eval_loss, pred = evaluate(x_val, y_val, custom, loss_fn_custom_model, device)

    # 예측 값 측정
    pred = nn.functional.softmax(pred, dim=1)
    pred = np.argmax(pred.cpu().numpy(), axis=1)
    acc = accuracy_score(y_val.numpy(), pred)

    tr_loss_list.append(tr_loss)
    val_loss_list.append(eval_loss)
    val_acc_list.append(acc)

    print(f"Epoch : {t + 1}, train loss : {tr_loss:.4f}, val loss : {eval_loss:.4f}, val accuracy : {acc:.4f}")

    if eval_loss > best_error:
        counter += 1
        print(f"----> early stopping status, best_loss : {best_error:.4f}, counter : {counter}")
    else:
        best_error = eval_loss
        counter = 0
    if counter >= patience:
        print("early stopping")
        torch.save(obj=custom, f="./cifarMode.pt")
        break

dl_learning_curve(tr_loss_list, val_loss_list, val_acc_list)

# 예측
_, pred = evaluate(x_test, y_test, custom, loss_fn_custom_model, device)
pred = nn.functional.softmax(pred, dim=1)
pred = np.argmax(pred.cpu().numpy(), axis=1)

# confusion matrix
cm = confusion_matrix(y_test.numpy(), pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
disp.plot()
plt.xticks(rotation=90)
plt.show()

# classification report
print('=' * 80)
print(f'Accuracy : {accuracy_score(y_test.numpy(), pred)}')
print('-' * 80)
print(classification_report(y_test.numpy(), pred, target_names=train_dataset.classes))
