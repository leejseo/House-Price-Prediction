# House-Price-Prediction
팀원: 유재민, 조영인, 최재민

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
