# 구글코랩에서 작성하였으며, 코드 구동 시에 csv 파일을 불러오는 형식
from google.colab import files
uploaded = files.upload()
my_data = "Data_Paper2_Case3_r.csv"

from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler

# 라이브러리 함수 불러오기
import tensorflow as tf
import numpy as np
np.set_printoptions(precision=2)

# 실행할 때마다 같은 결과를 출력하기 위해 설정하는 부분
np.random.seed(3)
tf.random.set_seed(3)

# 데이터 분류
dataset = np.loadtxt(my_data,delimiter=",")
scaler = MinMaxScaler()
dataset[:] = scaler.fit_transform(dataset[:])
X=dataset[:,0:102]
Y=dataset[:,102]

p_grid = {"C": [1, 10], "gamma": [.01, .1]}

svm = SVC(kernel="rbf")

inner_cv = KFold(n_splits=5, shuffle=True, random_state=3)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=3)

#'iid=False' 의 파라미터 설정을 없애니 오류 사라짐.
clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
clf.fit(X, Y)
best_estimator = clf.best_estimator_

cv_dic = cross_validate(clf, X, Y, cv=outer_cv, scoring=['accuracy'], return_estimator=False, return_train_score=True)
mean_val_score = cv_dic['test_accuracy'].mean()

print('nested_train_scores: ', cv_dic['train_accuracy'])
print('nested_val_scores:   ', cv_dic['test_accuracy'])
print('mean score:          {0:.2f}'.format(mean_val_score))