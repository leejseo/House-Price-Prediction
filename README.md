# House-Price-Prediction
팀원: 유재민, 조영인, 최재민

## Data Processing
먼저, 주어진 label 데이터의 빈도 분포를 확인하였고, log-normal 분포와 비슷함을 확인하였다.
![The actual transaction price distribution (before mapping)](/before.png "Figure 1: Before")

선형 모델이 정규 분포를 따르는 데이터에 대해 높은 성능을 보이기 때문에, label 데이터의 빈도 분포를 log(1+x) 함수를 이용하여 맵핑하였고, 이 값을 이용해 model을 fitting 하였다.
![The actual transaction price distribution (after mapping)](/after.png "Figure 2: After")

결손 데이터는 모두 0으로 채웠으며, 날짜형 feature의 데이터는 가장 오래된 해의 1월 1일로 부터 지난 날 수로 바꿨다.

## Environment
학습과 테스트를 진행한 환경은 다음과 같다:
  * OS: Windows 10 64bit
  * Processor: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz 2.21 GHz
  * RAM: 16.0 GB
  
## Performance
9,600여개의 데이터로 테스트한 결과 0.9487의 정확도를 보였다. 
정확도를 계산하는 식은 다음과 같다:  
  * (정확도) = 1 - 1/(데이터의 갯수) * (sum of (abs(실제 가격 - 예측한 가격)/(실제 가격)) for each data)

퍼포먼스 측정을 위해 대조군으로 활용한 방식에 비해 5.19%p 개선된 결과가 나왔다.

## Referensces
Stacked Regressions to predict House Prices from Kaggle
