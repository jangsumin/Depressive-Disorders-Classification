# 구글코랩에서 작성하였으며, 코드 구동 시에 csv 파일을 불러오는 형식
from google.colab import files
uploaded = files.upload()
my_data = "Data_Paper2_Case3.csv"

# MLP를 구현하기 위해 사용하는 keras 패키지 불러오기
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

# 라이브러리 함수 불러오기
import tensorflow as tf
import numpy as np

# One-Hot encoding을 위한 함수 불러오기
from sklearn.preprocessing import LabelEncoder

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분
np.random.seed(3)
tf.random.set_seed(3)

# 데이터 분류
dataset=np.loadtxt(my_data,delimiter=',')
X=dataset[:,2:102].astype(float)
Y=dataset[:,102]

# 결과(Y)가 '0','1'의 2진수로 이루어진 것이 아님. 
# '0','3'으로 이루어져 있기 때문에 입력값(X)가 활성화 함수의 일종인 Softmax 함수를
# 지나 총합이 1이고 원소가 2개인 벡터 형태를 띄게 되고, 교차 엔트로피를 지나 
# [1.,0.]이나 [0.,1.]으로 변하게 되는 One-Hot encoding 방식 채택 
e=LabelEncoder()
e.fit(Y)
y=e.transform(Y)
y_encoded=tf.keras.utils.to_categorical(y)

# 딥러닝 모델 구현
model=Sequential()
model.add(Dense(units=128,input_dim=100,activation='relu'))
model.add(Dense(units=32,activation='relu'))
model.add(Dense(units=8,activation='relu'))
model.add(Dense(units=4,activation='relu'))
model.add(Dense(units=2,activation='softmax'))

# 손실함수로 'categorical_crossentropy'를 채택하고, 'Adam'알고리즘 이용
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# 모델 훈련
model.fit(X,y_encoded,epochs=100,batch_size=10)
model.summary()