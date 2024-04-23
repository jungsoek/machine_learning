# Lec4. Multi-variable linear regression

## Basics of Tensorflow

session 이전 : 연산을 위한 구조를 만든다.

session 이후 : 텐서 흐름을 통해 연산을 수행

### placeholder, Variable

* placeholder(요소 자료형, shape=[얼마 들어올지 모르므로 None, feature 개수])

​	ex) tf.placeholder(tf.float32, shape[None, 3])

* Variable(tf.random_normal(행렬 연산 후의 shape))

​	ex) tf.Variable(tf.random_normal([3, 1]))

## 선형 회귀 및 다중 선형 회귀 비교

### 선형 회귀

#### Hypothesis

각각의 측정된 데이터와의 편차가 최소가 되는 선형 함수이다.

![image-20240409145344805](/home/oem/assets/image-20240409145344805.png)

#### Cost function

편차의 제곱(방향성을 제외하고 오직 함수와의 차이를 나타내기 위해)의 평균을 나타낸 함수이다.

![image-20240409145406172](/home/oem/assets/image-20240409145406172.png)

#### Gradient descent algorithm

1차 근사값 발견용 최적화 알고리즘이다. 기본 개념은 함수의 기울기(경사)를 구하고 경사의 반대방향으로 계속 이동시켜 극값에 이를 때까지 반복하는 것이다.

최적화할 함수 f(x)에 대하여, X0를 시작으로 그 다음으로 이동할 점인 Xi+1은 다음과 같이 계산된다.

![image-20240409154151985](/home/oem/assets/image-20240409154151985.png)

이때 감마는 이동할 거리를 조절하는 매개변수이다.

이 알고리즘의 수렴 여부는 f의 성질과 감마의 값에 따라 달라진다. 또한 이 알고리즘은 지역 최적해로 수렴한다.

따라서 구한 값이 전역적인 최적해라는 것을 보장하지 않으며 시작점 X0의 선택에 따라서 달라진다.

아래와 같이 local minimum으로 수렴을 하여도 Global minimum이 존재할 수도 있다.

![경사하강법(gradient descent) - 공돌이의 수학정리노트 ...](https://raw.githubusercontent.com/angeloyeo/angeloyeo.github.io/master/pics/2020-08-16-gradient_descent/pic5.png)

### 다중 선형 회귀

#### Hypothesis

![image-20240409145642075](/home/oem/assets/image-20240409145642075.png)

#### Cost function

![image-20240409145651214](/home/oem/assets/image-20240409145651214.png)

#### Matrix

![image-20240409145716948](/home/oem/assets/image-20240409145716948.png)

## Matrix multiplication

행렬 곱셈의 특징

* (2 by 3) * (3 by 2) = (2 by 2)
* 앞의 열 수와 뒤의 행 수가 일치해야 한다.

## Hypothesis using  matrix

![image-20240409125455805](/home/oem/assets/image-20240409125455805.png)

## Matrix

![image-20240409125525613](/home/oem/assets/image-20240409125525613.png)

### WX vs XW

머신러닝에서 WX + b 대신 XW + b를 사용하는 이유 : 

* W는 가중치 벡터이며 수학에서 벡터는 행이 아닌 열로 간주된다(d x 1).

* X는 데이터 모음이다.

  X는 행렬 n x d이다(n은 데이터 수이고 d는 특성의 수, 즉 열의 수이다).

위의 두 가지를 올바르게 곱하고 올바른 기능에 올바른 가중치를 사용하려면 XW + b를 사용해야 한다.

## Loading data from file

* 학습 데이터셋을 데이터베이스에서 엑셀로 받는다.
* 엑셀 파일을 그대로 사용하는 것이 아닌 .csv 확장자 파일로 변경 후 사용

# Lec5-1. Logistic (regression) classification

# : Cost function & gradient decent

## Logistic Regression

![image-20240409142319295](/home/oem/assets/image-20240409142319295.png)

## Regression(HCG)

### Hypothesis

![image-20240409145922803](/home/oem/assets/image-20240409145922803.png)

### Cost

![image-20240409145932129](/home/oem/assets/image-20240409145932129.png)

### Gradient decent

![image-20240409145943237](/home/oem/assets/image-20240409145943237.png)

## 사용 사례 : Classification(0, 1 encoding)

ex) 스팸 메일 분류, 페이스북 피드, 신용카드 사기적 사용 감지

: 모두 0과 1로 구분 가능

## Linear regression

ex) 공부 시간에 따른 시험 당락 여부

* 독립 변수 : 공부 시간
* 종속 변수 : 시험 성적
* Classification 결과 : 당락 여부

## Logistic Hypothesis

위의 예시처럼 로지스틱 회귀 모델은 여러 조건을 부합하여 결과를 0과 1로 출력하는 역할을 수행한다.

### Sigmoid

아래는 로지스틱 회귀에서 사용되는 시그모이드 함수인데, S자 모양을 띠고있다.

시그모이드 함수 식에 x값으로 0을 대입하면 정확이 0.5라는 결과로 계산

* x > 0, y는 0.5를 기준으로 긍정 결과
* x < 0, y는 0.5를 기준으로 부정 결과

![image-20240409150749285](/home/oem/assets/image-20240409150749285.png)

![image-20240409150728255](/home/oem/assets/image-20240409150728255.png)

시그모이드 함수를 사용하는 이유

: 사건이 발생하고, 발생하지 않는 결과를 선형으로 표현하게 됐을 때 문제가 발생하기 때문

![image-20240409152705518](/home/oem/assets/image-20240409152705518.png)

위와 같이 이상치의 추가가 분류 모델에 큰 영향을 미치는 문제점이 발생하므로 시그모이드 함수를 도입한 것이다.

그래프를 직선으로 표현하는 것 대신, 완만한 S자형 곡선으로 나타내어 위와 같은 상황의 문제를 방지한다.

# Lec5-2. Logistic (regression) classification cost function & gradient decent

## Cost 

선형 회귀의 이상적인 cost function은 아래와 같이 이차함수가 되지만,

![image-20240409154945992](/home/oem/assets/image-20240409154945992.png)

## Cost function

실제 측정치에 대한 cost function은 데이터의 산포와 자연상수 e에 의해 아래와 같이 울퉁불퉁한 모양을 가지게되며 여러 극값을 가지게 된다.

![image-20240409155655864](/home/oem/assets/image-20240409155655864.png)

![img](https://blog.kakaocdn.net/dn/cR0CLi/btq3mqb7MbD/CkEt1nsEqEKN8k4q44Jxck/img.png)

이러한 형태의 그래프에서 Gradient descent algorithm을 통한 학습을 하게되면 최솟값이 되는 지점을 잘못 찾을 수 있다(local minimum이 global minimum이 아니므로)(최솟값은 global minimum이다). 

따라서 이러한 자연상수e를 없애기 위해 그 역함수인 log를 사용한다.

## New cost function for logistic

![image-20240409155832355](/home/oem/assets/image-20240409155832355.png)

## understandig cost function

![image-20240409155839937](/home/oem/assets/image-20240409155839937.png)

## Cost function

![image-20240409155904685](/home/oem/assets/image-20240409155904685.png)

## Minimize cost - Gradient decent algorithm

![image-20240409155936623](/home/oem/assets/image-20240409155936623.png)

## Gradient decent algorithm

![image-20240409155947424](/home/oem/assets/image-20240409155947424.png)

# Lec6-1. Softmax classification

# Multinomial classification

## Logistic regression

![image-20240411124227546](/home/oem/assets/image-20240411124227546.png)

회귀분석의 개념을 이진 분류 문제로 확장한 Logistic Regression의 원리를 보면, 

![image-20240411124537602](/home/oem/assets/image-20240411124537602.png)

input에 대한 연산 결과를 0 ~ 1 사이의 확률값으로 표현하고, 

![image-20240411124451845](/home/oem/assets/image-20240411124451845.png)

![image-20240411124501146](/home/oem/assets/image-20240411124501146.png)

이를 두 가지 중에 하나로 결론을 내리는 방법이 logistic regression이다.

![image-20240411124654258](/home/oem/assets/image-20240411124654258.png)

이 원리를 이용해서 두 가지 중 하나가 아니라 여러 개의 결론을 내리는 것 역시 가능하다.

![image-20240411124800241](/home/oem/assets/image-20240411124800241.png)

![image-20240411124808288](/home/oem/assets/image-20240411124808288.png)

각각의 분류할 항목마다의 로지스틱 회귀를 여러 번 적용하는 것이며 이를 일반 행렬식으로 표현하면 다음과 같다.

![image-20240411124931494](/home/oem/assets/image-20240411124931494.png)

변수가 하나였을 때 로지스틱 회귀를 적용한 것과 마찬가지로 각각의 변수마다 로지스틱 회귀를 적용하여 0~1사이의 확률 값으로 표현하고 각각의 항목들(결론, classes)을 도출해낸다.

![image-20240411125235374](/home/oem/assets/image-20240411125235374.png)

각각의 항목들을 분류해내는데에는 각각의 score마다 확률표현식을 적용하여 확률값으로 표현한다.

![image-20240411125414566](/home/oem/assets/image-20240411125414566.png)

이것이 softmax classifier의 원리이며 위의 수식이 바로 그것이다.

일반적으로 Softmax classifier를 만들어내는 경우에는 정답 클래스를 one-hot encoding 방법으로 학습시킨다.

![image-20240411125742126](/home/oem/assets/image-20240411125742126.png)Softmax의 Cost function은 다음과 같다.

![image-20240411130027618](/home/oem/assets/image-20240411130027618.png)

cross-entropy를 cost function으로 사용하는 이유는 울퉁불퉁한 그래프를 완곡히 만들기 위해, 즉 미분의 편의성 때문이다. 미분의 편의성을 위하는 이유는 여기서도 경사하강법을 적용하기 때문이다.

![image-20240411130503165](/home/oem/assets/image-20240411130503165.png)

다음은 Cross-entropy를 cost function으로 사용하는 것이 적절한가를 판가름하는 예시이다.

![image-20240411130537866](/home/oem/assets/image-20240411130537866.png)

위의 그림과 수식을 보면 전혀 서로 완전히 상반되는 값([ [0], [1] ] / [ [1], [0] ])과 완전히 일치하는 값([ [1], [0] ] / [ [0], [1] ]), 이 두 값을 input으로 사용한 것을 볼 수 있다. 

완전히 상반되는 값은 cost가 무한히(혹은 아주 높게) 나와야하고, 완전히 일치하는 값은 cost가 0으로 수렴해야 한다. 결과를 확인하보면, cost값이 예상과 같이 나옴을 알 수 있다.

결론적으로 Cross-entropy를 cost function으로 사용하여도 적절하다는 것을 알 수 있다. 다음은 트레이닝 데이터셋을 Cross-entropy를 cost function으로 사용하는 함수에 사용하는 수식이다.

![image-20240411131114483](/home/oem/assets/image-20240411131114483.png)

# Lec 7-1. Application & Tips:

# Learning rate, data preprocessing, overfitting

다음은 cost function에 gradient descent(경사하강법)를 적용하는 모습이다.

![image-20240411131319390](/home/oem/assets/image-20240411131319390.png)

여기서 곱해지는 alpha 값에 주목해야 하는데 이는 learning_rate로, 이 값에 의해 함수의 경사를 얼마나 빠르게, 혹은 느리게, 띄엄띄엄, 촘촘히 하강하는지가 결정된다.

alpha 값을 크게 책정하면 빠르게 cost의 최소값을 찾을 수 있으나 발산해서 전혀 엉뚱한 결과가 나올 수도 있다.

![image-20240411142624543](/home/oem/assets/image-20240411142624543.png)

너무 촘촘히 책정하면 cost의 최소값을 찾는 속도가 느려지고 연산양이 많아 컴퓨터에 많은 부하가 주게된다. 따라서 alpha(learning_rate) 값은 적정하게 책정해야 한다.

![image-20240411131558568](/home/oem/assets/image-20240411131558568.png)

## Overfitting(과적합)

과적합은 과한 학습 혹은 필요이상으로 많은 특성에 의해 이상치의 데이터까지 평균치의 class로 분류하는 경우를 말한다.

혹은 경계를 넘나드는 데이터를 전혀 다른 class로 분류하는 것까지 과적합으로 본다.

일반적으로 학습 데이터는 실제 데이터의 부분 집합이므로 학습 데이터에 대해서는 오차가 감소하지만 실제 데이터에 대해서는 오차가 증가하게 된다.

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Overfitting.svg/250px-Overfitting.svg.png)

아래의 예시를 보면 모델의 경계를 넘나드는 이상치 데이터를 확인할 수 있는데 이상치를 덜어내는 것이 아니라 이상치 데이터까지 학습모델에 적용시켜 이상치 데이터까지 평균치의 class로 분류하는 것을 볼 수 있다.

![image-20240411143339431](/home/oem/assets/image-20240411143339431.png)

과적합의 대표적인 사례로는 다음과 같다.

* 한 data set에만 지나치게 최적화된 상태
* 학습 데이터에 대해서는 오차가 감소하지만 실제 데이터에 대해서는 오차가 증가하는 경우

즉 과적합은 학습 데이터에 대해 과하게 학습하여 실제 데이터에 대한 오차가 증가할 경우 발생한다.

과적합 현상을 방지하려면 

* 더 많은 데이터셋에 대한 학습, 

* 특성 수 줄이기, 

* 정칙화(regularization) 등을 수행한다.

  여기서 정칙화란 W(weight)가 너무 큰 값들을 가지지 않도록 하는 것이다.

  W가 너무 큰 값을 가지게 되면 구불구불한 형태의 함수가 만들어지는데, Regularization은 이런 모델의 복잡도를 낮추기 위한 방법이다.

  Regularization은 단순하게 cost function을 작아지는 쪽으로 학습하면 특정 가중치 값들이 커지면서 결과를 나쁘게 만들기 때문에 cost function을 바꾼다.

  ![image-20240411144829023](/home/oem/assets/image-20240411144829023.png)

과적합(혹은 과대적합)의 반대의 의미로 과소적합(underfitting)이 있는데, 과소적합은 기계 학습에서 통계 모형의 능력 부족으로 학습 데이터를 충분히 설명하지 못하도록 부족하게 학습된 것을 말한다.

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Underfitted_Model.png/250px-Underfitted_Model.png)

# Lec 7-2. Application & Tips:

# Learning and test data sets

## Train, validation and test sets

![image-20240411164447288](/home/oem/assets/image-20240411164447288.png)

Train set : 모델을 학습하기 위한 dataset이다.

Validation set : 학습이 이미 완료된 모델을 검증하기위한 dataset이다.

Test set : 모델의 성능을 평가하기 위한 dataset이다.

![image-20240411165528793](/home/oem/assets/image-20240411165528793.png)

위에서 언급되지는 않았으나 Validation set과 Test set에 대해서 살펴본다면, Validation set과 Test set의 공통점은 학습을 시키지 않고 모델을 검증하고 평가한다는 것이다.

그러나 이 둘 간의 차이가 있는데,

Validation set의 경우는 학습을 시키지는 않지만 학습에 관여한다.

Test set의 경우는 학습에 전혀 관여하지 않고 오직 '최종 성능'을 평가하기 위해 쓰인다.

Train set으로 학습을 할 때 너무 높은 epoch로 학습시키면 overfitting의 문제가 있다.

때문에 적정한 epoch를 찾아야 한다. 이때 사용하는 것이 validation set이다.

validation set은 train set에 의한 epoch뿐만 아니라 다른 hyperparameter, hidden layer를 조정할때도 사용할 수 있다. 예를 들어, learning rate와 hidden layer를 조금씩 변형해가면서 validation set에 대한 accuracy를 보면서 적정한 hyperparameter, hidden layer를 결정하는 것이다.

Validation set에 대해서 accuracy는 아주 중요하다. 

overfitting에 빠지지 않고, unseen data에 대한 좋은 성능을 발휘해야하기 때문이다.

따라서 최종 성능 평가를 위한 test 이전에 validation으로 어떤 모델을 어떤 epoch로 학습시킬때 unseen data에 대해서 좋은 성능을 보이는 모델이 무엇인지 파악해야 한다.



# Lec7. CNN(Convolution Neural Network)

## 컨볼루션 신경망 개념 - 컨볼루션, 풀링

