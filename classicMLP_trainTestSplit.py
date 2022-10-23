# 구글코랩에서 작성하였으며, 코드 구동 시에 csv 파일을 불러오는 형식
from google.colab import files
uploaded = files.upload()
my_data = "Data_Paper2_Case3_r.csv"

# MLP를 구현하기 위해 사용하는 keras 패키지 불러오기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras import optimizers
from sklearn.model_selection import train_test_split

# 라이브러리 함수 불러오기
import tensorflow as tf
import numpy as np

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분
np.random.seed(3)
tf.random.set_seed(3)

# 데이터 분류
dataset=np.loadtxt(my_data,delimiter=",")
X=dataset[:,0:102]
Y=dataset[:,102]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 3)

# 딥러닝 모델 구현
model=Sequential()
model.add(Dense(units=128,input_dim=102,activation='relu'))
model.add(Dense(units=32,activation='relu'))
model.add(Dense(units=8,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

# 손실함수로 'binary_crossentropy'를 채택하고, 'Adam'알고리즘 이용
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# 모델 훈련
model.fit(X_train,Y_train,batch_size=5,epochs=100)
model.summary()