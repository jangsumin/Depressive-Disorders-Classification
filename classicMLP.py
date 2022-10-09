# 구글 코랩에서 파일 읽기
# 구글 코랩에서 진행하는 것이 수월한데,
# 머신 러닝을 활용하기 위한 패키지 혹은 라이브러리를 따로 설치할 필요가 없습니다.
# 우울장애 분류를 위한 csv 파일은 비공개입니다.
from google.colab import files
uploaded = files.upload()
my_data = "Data_Paper2_Case3_r.csv"

# VSCode 상에서 파일 읽기
# my_data = open('/Users/jangsumin/Desktop/machine_learning/Data_Paper2_Case3_r.csv')

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

#텐서플로 불러오기
import tensorflow as tf
import numpy as np
import pandas as pd

seed=0
np.random.seed(seed)
tf.random.set_seed(3)

dataset=np.loadtxt(my_data,delimiter=',')
X=dataset[:,2:102]
Y=dataset[:,102]

model=Sequential()
model.add(Dense(units=128,input_dim=100,activation='relu'))
model.add(Dense(units=32,activation='relu'))
model.add(Dense(units=8,activation='relu'))
model.add(Dense(units=4,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X,Y,epochs=100,batch_size=10)
model.summary()