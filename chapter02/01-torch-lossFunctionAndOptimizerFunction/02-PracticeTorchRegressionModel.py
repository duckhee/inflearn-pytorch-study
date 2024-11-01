import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam


