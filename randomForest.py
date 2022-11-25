# 구글코랩에서 작성하였으며, 코드 구동 시에 csv 파일을 불러오는 형식
from google.colab import files
uploaded = files.upload()
my_data = "Data_Paper2_Case3_r.csv"

# MLP를 구현하기 위해 사용하는 keras 패키지 불러오기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras import optimizers
from numpy import mean
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 라이브러리 함수 불러오기
import tensorflow as tf
import numpy as np
import pandas as pd

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분
np.random.seed(1)
tf.random.set_seed(1)

# 데이터 분류
dataset=np.loadtxt(my_data,delimiter=",")
scaler = MinMaxScaler()
dataset[:] = scaler.fit_transform(dataset[:])
X=dataset[:,0:102]
Y=dataset[:,102]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)
inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)

params = {
    'n_estimators' : [100], 'max_depth' : [4, 6, 8, 10],
    'min_samples_leaf' : [8, 12, 18], 'min_samples_split' : [8, 16, 20]
}

rf_clf = RandomForestClassifier(random_state = 1, n_jobs= -1)
grid_cv = GridSearchCV(rf_clf, param_grid=params, cv=inner_skf, n_jobs= -1)
grid_cv.fit(X_train, Y_train)

print('최적 하이퍼 파라미터:\n', grid_cv.best_params_)
print('최고 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))