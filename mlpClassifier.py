# 구글코랩에서 작성하였으며, 코드 구동 시에 csv 파일을 불러오는 형식
from google.colab import files
uploaded = files.upload()
my_data = "Data_Paper2_Case3_r.csv"

# MLP를 구현하기 위해 사용하는 sklearn 패키지 불러오기
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
#from sklearn.ensemble import RandomForestClassifier
from numpy import mean, std, max
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# 라이브러리 함수 불러오기
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# 데이터 분류 및 전처리
dataset=np.loadtxt(my_data,delimiter=",")
#scaler = MinMaxScaler()
#dataset[:] = scaler.fit_transform(dataset[:])
#dataset[:] = fit_transform(dataset[:])
X=dataset[:,0:102]
Y=dataset[:,102]

# MLPClassifier를 이용하기 위한 하이퍼 파라미터
parameter_space = {'hidden_layer_sizes': [(128,128,128), (256,256,256), (512,512,512)],
     'activation': ['relu'],
     'solver': ['adam'],
     'max_iter': [1000],
     'alpha': [0.1],
     'learning_rate': ['constant']}

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분
#np.random.seed(0)
#tf.random.set_seed(0)
r_list = np.random.randint(1, 100, size = 2)

for r in r_list:

  # fold 수 결정
  outer_skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=r)
  inner_skf = StratifiedKFold(n_splits=4, shuffle = True, random_state=r)
  #outer_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=r)
  #inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=r)

  # MLPClassifier를 mlp 변수로 받기
  mlp = MLPClassifier(random_state = r)

  outer_results = list()

  # X,Y를 train(학습)영역과 test(테스트)영역으로 분리 후, X 영역의 학습 데이터만 정규화 진행
  for train, test in outer_skf.split(X, Y):
    X_train, X_test = X[train, :], X[test, :]
    Y_train, Y_test = Y[train], Y[test]
  
    mm_scaler = MinMaxScaler()
    X_train_scaled = mm_scaler.fit_transform(X_train)
    X_test_scaled = mm_scaler.transform(X_test)
  
    # GridSearchCV 이용, 검증 진행, 검증을 하는 동안에 제일 좋은 하이퍼 파라미터의 조합을 변수로 지정
    # 그 조합을 가지고 예측 진행, 그때의 정확도를 sklearn에서 제공하는 accuracy_score함수를 이용하여 측정
    search = GridSearchCV(mlp, param_grid = parameter_space , scoring = 'accuracy', cv = inner_skf, refit = True )
    result = search.fit(X_train_scaled, Y_train)
    best_model = result.best_estimator_
    ypred = best_model.predict(X_test_scaled)
    acc = accuracy_score(Y_test, ypred)
    outer_results.append(acc)
    print('>acc=%.4f, est=%.4f, cfg=%s' % (acc, result.best_score_, result.best_params_))

  # 모델을 예측에 사용했을 때의 정확도 
  results.append(mean(outer_results))
  print('Accuracy: %.4f (%.4f) , (random_state = %d)' % (mean(outer_results), std(outer_results), r))