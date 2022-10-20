# Depressive_disorders_classification
organize 'classification of depressive disorders utilizing machine learning' study which I have done since 2021.08
- - -
* 학부 수준의 논문을 작성하기 위한 우울장애 분류 연구를 21년 8월부터 진행했습니다.
* 22-10-06 부로 연구를 마무리하였습니다.
* 연구한 내용들을 정리해서 업로드하며, 논문 작성을 마무리합니다.
* Google Colab에서 소스 코드들을 작성하고, 실행 결과를 얻어냈습니다.
* 우울장애 분류작업을 위해 사용한 csv 형식의 파일은 비공개입니다.
- - -
### 21.08
* 매우 기본적인 딥러닝 모델에 우울증 분류 데이터가 적용되는지 확인하였습니다. ➡️ `classicMLP.py`
* One-Hot encoding 방식을 사용하여 데이터의 Y Label(분류 결과)이 0과 3으로 나누어져 있는 부분을 고려하였습니다. 하지만, 분류 모델을 검증하기 위한 테스트 데이터 없이 학습(train)만 진행했기 때문에 유효한 결과를 얻지 못했습니다. ➡️ `classicMLP_LabelEncoder.py`
