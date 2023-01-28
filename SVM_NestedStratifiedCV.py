# 구글코랩에서 작성하였으며, 코드 구동 시에 csv 파일을 불러오는 형식
from google.colab import files
uploaded = files.upload()
my_data = "Data_Paper2_Case3_r.csv"

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# 라이브러리 함수 불러오기
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분
np.random.seed(3)
tf.random.set_seed(3)

# 데이터 분류
dataset = np.loadtxt(my_data,delimiter=",")
X = dataset[:,0:102]
Y = dataset[:,102]

p_grid = {"C": [1, 10], "gamma": [.01, .1]}
svm = SVC(kernel = "rbf")

results = list()

for r in range(10):

    inner_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=r)
    outer_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=r)

    for train, test in outer_skf.split(X, Y):
        X_train, X_test = X[train, :], X[test, :]
        Y_train, Y_test = Y[train], Y[test]
  
        mm_scaler = MinMaxScaler()
        X_train_scaled = mm_scaler.fit_transform(X_train)
        X_test_scaled = mm_scaler.transform(X_test)

        outer_results = list()
    
        #'iid=False' 의 파라미터 설정을 없애니 오류 사라짐.
        clf = GridSearchCV(estimator=svm, param_grid=p_grid, scoring='accuracy', cv=inner_skf, refit=True)
        result = clf.fit(X_train_scaled, Y_train)
        best_model = result.best_estimator_
        ypred = best_model.predict(X_test_scaled)
        acc = accuracy_score(Y_test, ypred)
        outer_results.append(acc)
        print('>acc=%.4f, est=%.4f, cfg=%s' % (acc, result.best_score_, result.best_params_))

    # 모델을 예측에 사용했을 때의 정확도 
    results.append(mean(outer_results))
    print('Accuracy: %.4f (%.4f) , (random_state = %d)' % (mean(outer_results), std(outer_results), r))
