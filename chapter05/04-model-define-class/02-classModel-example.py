import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from torchsummary import summary


class CustomCMM(nn.Module):
    def __init__(self, n_classes):
        super(CustomCMM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=8 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


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
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # 텐서 데이터 셋으로 합치기
    train_data_set = TensorDataset(x_train_tensor, y_train_tensor)

    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)

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
    data_size = len(data_loader.dataset)
    batch_size = len(data_loader)
    train_loss = 0
    model.train()

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss

        # back propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    train_loss /= batch_size
    return train_loss.item()


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
    plt.plot(epochs, val_acc_list, label='Accuracy', marker='.')

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid()
    plt.show()


device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

print(f"using device : {device}")

# 학습 데이터 가져오기 -> dataset 에 있는 Fashion data
train_data_set = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data_set = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=ToTensor()
)

# data 와 레이블 추출
train_data = train_data_set.data.numpy() / 255
train_labels = train_data_set.targets.numpy()
test_data = test_data_set.data.numpy() / 255
test_labels = test_data_set.targets.numpy()

# 데이터 샘플링 및 층화 추출
x_train, _, y_train, _ = train_test_split(train_data, train_labels, test_size=40000, random_state=10,
                                          stratify=train_labels)
x_val, x_test, y_val, y_test = train_test_split(test_data, test_labels, test_size=5000, random_state=10,
                                                stratify=test_labels)

# 이미지 데이터이기 때문에 4채널로 만들어준다.
x_train = x_train.reshape(20000, 1, 28, 28)
x_val = x_val.reshape(5000, 1, 28, 28)
x_test = x_test.reshape(5000, 1, 28, 28)

# tensor 로 변환
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
x_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 학습 시 이용할 TensorDataSet 생성
train_tensorDS = TensorDataset(x_train, y_train)

classes = train_data_set.classes
print(f"classes list : {classes}")

# 학습 시 사용할 DataLoader
batch_size = 64
train_data_loader = DataLoader(train_tensorDS, batch_size=batch_size)

# 모델에 대한 예측할 목록 갯수
model_classes_number = len(classes)

model = CustomCMM(model_classes_number)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

summary(model, input_size=(1, 28, 28))
model.to(device)

epochs = 10
tr_loss_list, val_loss_list, val_acc_list = [], [], []

for t in range(epochs):
    tr_loss = train(train_data_loader, model, loss_fn, optimizer, device)
    val_loss, pred = evaluate(x_val, y_val, model, loss_fn, device)

    # accuracy 측정
    pred = nn.functional.softmax(pred, dim=1)
    pred = np.argmax(pred.cpu().numpy(), axis=1)
    acc = accuracy_score(y_val.numpy(), pred)

    # 리스트에 추가
    tr_loss_list.append(tr_loss)  # train - CrossEntropy
    val_loss_list.append(val_loss)  # val - CrossEntropy
    val_acc_list.append(acc)  # val - Accuracy

    print(f"Epoch {t + 1}, train loss : {tr_loss:.4f}, val loss : {val_loss:.4f}, val acc : {acc:.4f}")

# 학습곡선
dl_learning_curve(tr_loss_list, val_loss_list, val_acc_list)

# 예측
_, pred = evaluate(x_test, y_test, model, loss_fn, device)
pred = nn.functional.softmax(pred, dim=1)
pred = np.argmax(pred.cpu().numpy(), axis=1)

# confusion matrix
cm = confusion_matrix(y_test.numpy(), pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot()
plt.xticks(rotation=90)
plt.show()

# classification report
print('=' * 80)
print(f'Accuracy : {accuracy_score(y_test.numpy(), pred)}')
print('-' * 80)
print(classification_report(y_test.numpy(), pred, target_names=classes))
