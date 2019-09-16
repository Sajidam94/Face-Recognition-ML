# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


data = pd.read_csv(os.getcwd()+r"\dataset\face_data.csv")

X = data.iloc[:, 0].values

y = data.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

