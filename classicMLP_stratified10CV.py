# 구글코랩에서 작성하였으며, 코드 구동 시에 csv 파일을 불러오는 형식
from google.colab import files
uploaded = files.upload()
my_data = "Data_Paper2_Case3_r.csv"

# MLP를 구현하기 위해 사용하는 keras 패키지 불러오기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras import optimizers
from numpy import mean
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

# 라이브러리 함수 불러오기
import tensorflow as tf
import numpy as np
import pandas as pd

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분
np.random.seed(3)
tf.random.set_seed(3)

# 데이터 분류
dataset=np.loadtxt(my_data,delimiter=",")
scaler = MinMaxScaler()
dataset[:] = scaler.fit_transform(dataset[:])
X=dataset[:,0:102]
Y=dataset[:,102]

n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=3)

# 딥러닝 모델 구현
accuracy = []
for train, test in skf.split(X, Y):
  model=Sequential()
  model.add(Dense(units=128,input_dim=102,activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(units=32,activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(units=8,activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(units=1,activation='sigmoid'))
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

  # 모델 훈련
  model.fit(X[train], Y[train], epochs=100, batch_size=10)
  k_accuracy = model.evaluate(X[test], Y[test])[1]
  accuracy.append(k_accuracy)

print("\n %.f fold accuracy:" % n_fold, accuracy)
print("Accuracy: %.4f " % (mean(accuracy)))