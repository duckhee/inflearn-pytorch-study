import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import *

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from torchsummary import summary

import random as rd
import cv2

from os import getcwd, pardir, path


def get_image_path(name: str) -> str:
    """
    :param name:
    :return:
    """
    execute_path = getcwd()
    get_parent_dir = path.abspath(path.join(execute_path, pardir))
    image_folder = path.join(get_parent_dir, 'images')
    return path.join(image_folder, name)


def make_dataset(x_train, x_val, y_train, y_val, batch_size=32):
    """
    :param x_train:
    :param x_val:
    :param y_train:
    :param y_val:
    :param batch_size:
    :return:
    """

    x_train_tensor = torch.tensor(x_train, dtype=torch.floag32)
    x_val_tensor = torch.tensor(x_val, dtype=torch.floag32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

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

    tr_loss = 0

    model.train()

    for batch, (X, y) in enumerate(data_loader):
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


def dl_learning_curve(tr_loss_list, val_loss_list):
    """
    :param tr_loss_list:
    :param val_loss_list:
    :return:
    """
    epochs = list(range(1, len(tr_loss_list) + 1))

    plt.plot(epochs, tr_loss_list, label='Training', marker='.')
    plt.plot(epochs, val_loss_list, label='Validation', marker='.')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.show()


def predicated_custom(model, device):
    image_name = str(input("resource name input.. >> "))
    print(f"image name : {image_name}")
    user_image_path = get_image_path(image_name)
    custom_image = cv2.imread(user_image_path, cv2.IMREAD_GRAYSCALE)

    if custom_image is None:
        print("Load Image Failed...")
        exit()

    # show input image
    cv2.imshow("custom image", custom_image)
    cv2.waitKey(0)

    resize_custom_image = cv2.resize(255 - custom_image, (28, 28))
    resize_custom_image = resize_custom_image.reshape(1, 1, 28, 28)
    resize_custom_image = resize_custom_image / 255

    new_custom_data_tensor = torch.tensor(resize_custom_image, dtype=torch.float32)

    model.eval()

    with torch.no_grad():
        new_custom_data_tensor = new_custom_data_tensor.to(device)
        pred = model(new_custom_data_tensor)
        pred = nn.functional.softmax(pred, dim=1)
        pred = np.argmax(pred.cpu().numpy(), axis=1)
    return pred[0]


device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

print(f"using device : {device}")

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),  # 픽셀값을 [0,1] 사이로 정규화하고 텐서로 변환
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
batch_size = 64
train_data_loader = DataLoader(train_data, batch_size=batch_size)

for X, y in train_data_loader:
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape} data type : {y.dtype}")
    break

x_val, x_test = test_data.data[:5000], test_data.data[5000:]
y_val, y_test = test_data.targets[:5000], test_data.targets[5000:]

x_val = x_val / 255
x_test = x_test / 255

x_val = x_val.view(5000, 1, 28, 28)
x_test = x_test.view(5000, 1, 28, 28)

print(f"x val shape : {x_val.shape}, x test shape : {x_test.shape}")

n_class = 10

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    nn.ReLU(),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    nn.Flatten(),
    nn.Linear(64 * 7 * 7, out_features=128),
    nn.ReLU(),
    nn.Linear(128, out_features=n_class),
).to(device)

# summary 를 사용을 할 때에는 cuda 또는 cpu만 사용이 가능하다.
if device == 'cpu' or device == 'cuda':
    print(f"{summary(model, input_size=(1, 28, 28), device=device)}")
elif device == 'mps':
    print(f"{summary(model.cpu(), input_size=(1, 28, 28), device='cpu')}")
    model.to(device)
print("*" * 100)

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

epochs = 5

tr_loss_list, val_loss_list = [], []

for i in range(epochs):
    tr_loss = train(train_data_loader, model, loss_fn, optimizer, device)
    val_loss, _ = evaluate(x_val, y_val, model, loss_fn, device)
    tr_loss_list.append(tr_loss)
    val_loss_list.append(val_loss)

    print(f"epoch : {i + 1}, train loss : {tr_loss:4f}, validation loss : {val_loss:4f}")

dl_learning_curve(tr_loss_list, val_loss_list)

_, pred = evaluate(x_test, y_test, model, loss_fn, device)

# 예측 값을 확인하기 위해서 softmax로 변환
pred = nn.functional.softmax(pred, dim=1)
# 평균 최대 값을 찾기 위해서 예측 값을 CPU로 가져와서 numpy 로 변환을 해준다.
pred = np.argmax(pred.cpu().numpy(), axis=1)

print(f"predication : {pred}")

# confusion metrix 출력
cm = confusion_matrix(y_test.cpu().numpy(), pred)

# confusion metrix display
display_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_data.classes)
display_cm.plot()
plt.xticks(rotation=90)
plt.show()

# classification report
print(f"accuracy : {accuracy_score(y_test.cpu().numpy(), pred)}")
print("-" * 100)
print(f"classification report : {classification_report(y_test.cpu().numpy(), pred)}")

idx = (y_test.numpy() != pred)

x_test_wr = x_test[idx]
y_test_wr = y_test[idx]
pred_wr = pred[idx]

x_test_wr = x_test_wr.reshape(-1, 28, 28)
print(f"x_test shape : {x_test_wr.shape}")

idx = rd.sample(range(x_test_wr.shape[0]), 16)

x_temp = x_test_wr[idx]
y_temp = y_test_wr[idx]
pred_temp = pred_wr[idx]

plt.figure(figsize=(10, 10))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_temp[i], cmap=plt.cm.binary)
    plt.xlabel(f'actual : {y_temp[i]}, predicted : {pred_temp[i]}')
plt.tight_layout()
plt.show()

# 사용자가 작성한 이미지 예측 -> 크기는 100x100
image_path = get_image_path("handWriteSeven.png")

if image_path == "" or image_path is None:
    print("no image")
    exit()

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Load Image Failed...")
    exit()

cv2.imshow("image", image)

print(f"image shape : {image.shape}")

# 이미지 크기 조절
resize_image = cv2.resize(255 - image, (28, 28))
cv2.imshow("resize_image", resize_image)

print(f"resize_image shape : {resize_image.shape}")

# (1, 1, 28, 28)로 모양 재정의
new_data = resize_image.reshape(1, 1, 28, 28)

# 스케일링
new_data = new_data / 255

# tensor 로 변환
new_data_tensor = torch.tensor(new_data, dtype=torch.float32)

# model 평가 모드 변경
model.eval()

with torch.no_grad():
    new_data_tensor = new_data_tensor.to(device)
    pred = model(new_data_tensor)
    pred = nn.functional.softmax(pred, dim=1)
    pred = np.argmax(pred.cpu().numpy(), axis=1)

# 평가 결과 값 확인
print(f"predication : {pred}")

result = predicated_custom(model, device)
print(f"predication result : {result}")
