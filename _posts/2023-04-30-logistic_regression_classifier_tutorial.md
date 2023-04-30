---
layout: single
title:  "6차 과제에 대한 내용입니다."
categories: coding
tag: [python, blog]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


<a class="anchor" id="0"></a>

# **Python을 사용한 로지스틱 회귀 분류기 튜토리얼**



이 커널에서는 Python과 Scikit-Learn으로 Logistic Regression을 구현합니다. 내일 호주에 비가 올지 여부를 예측하기 위해 로지스틱 회귀 분류기를 구축하고, 로지스틱 회귀를 사용하여 이진 분류 모델을 교육한다.


# **1. Logistic Regression 소개** <a class="anchor" id="1"></a>



데이터 과학자가 새로운 분류 문제를 접할 때 가장 먼저 떠오를 수 있는 알고리즘은 **Logistic Regression** 이다. 이산 클래스 집합에 대한 관찰을 예측하는 데 사용되는 지도 학습 분류 알고리즘이다. 실제로 관측치를 여러 범주로 분류하는 데 사용된다. 따라서 출력은 본질적으로 이산적이다. **Logistic Regression**는 **Logit Regression**라고도 한다. 분류 문제를 해결하는 데 사용되는 가장 간단하고 직관적이며 다양한 분류 알고리즘 중 하나이다.


# **2. Logistic Regression 직관** <a class="anchor" id="2"></a>



통계에서 **Logistic Regression model**은 주로 분류 목적으로 사용되는 널리 사용되는 통계 모델이다. 이는 일련의 관찰이 주어지면 로지스틱 회귀 알고리즘이 이러한 관찰을 두 개 이상의 이산 클래스로 분류하는 데 도움이 된다는 것을 의미한다. 따라서 대상 변수는 본질적으로 이산적이다.





로지스틱 회귀 알고리즘은 다음과 같이 작동한다.



**선형 방정식 구현**





로지스틱 회귀 알고리즘은 응답 값을 예측하기 위해 독립 또는 설명 변수가 있는 선형 방정식을 구현하여 작동한다. 예를 들어 공부한 시간과 시험에 합격할 확률의 예를 고려한다. 여기서 학습시간은 설명변수로 x1로 표시한다. 시험에 합격할 확률은 응답 또는 목표 변수이며 z로 표시된다.





하나의 설명 변수(x1)와 하나의 응답 변수(z)가 있는 경우 선형 방정식은 수학적으로 다음 방정식으로 주어집니다.



     z = β0 + β1x1



여기서 계수 β0 및 β1은 모델의 매개변수입니다.





설명 변수가 여러 개인 경우 위의 방정식은 다음과 같이 확장될 수 있다.



     z = β0 + β1x1+ β2x2+……..+ βnxn

    

여기서 계수 β0, β1, β2 및 βn은 모델의 매개변수이다.



따라서 예측 응답 값은 위의 방정식으로 주어지며 z로 표시된다.


**sigmoid 함수**



z로 표시된 이 예측 응답 값은 0과 1 사이의 확률 값으로 변환된다. 예측 값을 확률 값에 매핑하기 위해 시그모이드 함수를 사용한다. 그런 다음 이 시그모이드 함수는 모든 실제 값을 0과 1 사이의 확률 값으로 매핑한다.



기계 학습에서 시그모이드 함수는 예측을 확률에 매핑하는 데 사용된다. 시그모이드 함수는 S자 모양의 곡선을 가진다. 시그모이드 곡선이라고도 한다.



Sigmoid 함수는 Logistic 함수의 특별한 경우이다. 다음 수학 공식으로 제공됩니다.



그래픽으로 시그모이드 함수를 다음 그래프로 나타낼 수 있다.


Sigmoid Function



![Sigmoid Function](https://miro.medium.com/max/970/1*Xu7B5y9gp0iL5ooBj7LtWw.png)


**결정 경계**



시그모이드 함수는 0과 1 사이의 확률 값을 반환한다. 그런 다음 이 확률 값은 "0" 또는 "1"인 개별 클래스에 매핑된다. 이 확률 값을 불연속 클래스(합격/불합격, 예/아니오, 참/거짓)에 매핑하기 위해 임계값을 선택한다. 이 임계값을 결정 경계라고 합니다. 이 임계값 이상에서는 확률 값을 클래스 1에 매핑하고 그 아래에서는 값을 클래스 0에 매핑한다.



수학적으로 다음과 같이 표현할 수 있습니다.



p ≥ 0.5 => class = 1



p < 0.5 => class = 0



일반적으로 결정 경계는 0.5로 설정된다. 따라서 확률 값이 0.8(> 0.5)이면 이 관찰을 클래스 1에 매핑한다. 마찬가지로 확률 값이 0.2(< 0.5)이면 이 관찰을 클래스 0에 매핑한다. 이는 그래프에 표시된다.


![Decision boundary in sigmoid function](https://ml-cheatsheet.readthedocs.io/en/latest/_images/logistic_regression_sigmoid_w_threshold.png)


**예측하기**



이제 우리는 로지스틱 회귀에서 시그모이드 함수와 결정 경계에 대해 알게 되었다. 시그모이드 함수와 결정 경계에 대한 지식을 사용하여 예측 함수를 작성할 수 있다. 로지스틱 회귀의 예측 함수는 관찰이 긍정적일 확률(예 또는 참)을 반환한다. 이것을 class 1이라고 하고 P(class = 1)로 표시합니다. 확률이 1에 가까우면 관찰이 class 1에 있고 그렇지 않으면 class 0이라는 모델에 대해 더 확신할 수 있다.


# **3. 로지스틱 회귀의 가정** <a class="anchor" id="3"></a>



로지스틱 회귀 모델에는 몇 가지 주요 가정이 필요하다. 이들은 다음과 같다:



1. 로지스틱 회귀 모델에서는 종속 변수가 본질적으로 이진, 다항 또는 순서여야 한다.



2. 관찰이 서로 독립적이어야 한다. 따라서 관찰은 반복 측정에서 나오지 않아야 한다.



3. 로지스틱 회귀 알고리즘은 독립 변수 간의 다중 공선성이 거의 또는 전혀 필요하지 않다. 이는 독립 변수가 서로 너무 높은 상관관계를 가지지 않아야 함을 의미한다.



4. 로지스틱 회귀 모델은 독립 변수와 로그 확률의 선형성을 가정한다.



5. 로지스틱 회귀 모델의 성공 여부는 샘플 크기에 따라 다르다. 일반적으로 높은 정확도를 달성하려면 큰 샘플 크기가 필요하다.


# **4. 로지스틱 회귀 유형** <a class="anchor" id="4"></a>



로지스틱 회귀 모델은 대상 변수 범주에 따라 세 그룹으로 분류할 수 있다. 이 세 그룹은 아래에 설명되어 있다.



### 이진 로지스틱 회귀



이진 로지스틱 회귀에서 대상 변수에는 두 가지 가능한 범주가 있다. 카테고리의 일반적인 예는 예 또는 아니오, 좋음 또는 나쁨, 참 또는 거짓, 스팸 또는 스팸 없음, 합격 또는 불합격이다.





### 다항 로지스틱 회귀



다항 로지스틱 회귀에서 대상 변수에는 특정 순서가 아닌 세 개 이상의 범주가 있다. 따라서 세 개 이상의 명목 범주가 있다. 예에는 사과, 망고, 오렌지 및 바나나와 같은 과일 범주 유형이 포함된다.





### 순서형 로지스틱 회귀



순서형 로지스틱 회귀분석에서 대상 변수에는 세 개 이상의 순서 범주가 있다. 따라서 범주와 관련된 고유한 순서가 있다. 예를 들어, 학생의 성과는 나쁨, 보통, 양호 및 우수로 분류할 수 있다.


# **5. 라이브러리 가져오기** <a class="anchor" id="5"></a>



```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
```

데이터를 적재하고 정재하는데 필요한 라이브러리를 가져왔다.



```python
import warnings

warnings.filterwarnings('ignore')
```

# **6. 데이터 가져오기** <a class="anchor" id="6"></a>








```python
data = '/content/sample_data/weatherAUS.csv'

df = pd.read_csv(data)
```

# **7. 탐색적 데이터 분석** <a class="anchor" id="7"></a>



이제 데이터에 대한 인사이트를 얻기 위해 데이터를 탐색하겠습니다.



```python
# view dimensions of dataset

df.shape
```

<pre>
(145460, 23)
</pre>
데이터 세트에 145460개의 인스턴스와 23개의 변수가 있음을 알 수 있다.



```python
# preview the dataset

df.head()
```


  <div id="df-dddda83a-a0ab-446c-92d4-9c5a8a4885af">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>...</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday</th>
      <th>RainTomorrow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008-12-01</td>
      <td>Albury</td>
      <td>13.4</td>
      <td>22.9</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>44.0</td>
      <td>W</td>
      <td>...</td>
      <td>71.0</td>
      <td>22.0</td>
      <td>1007.7</td>
      <td>1007.1</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>21.8</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008-12-02</td>
      <td>Albury</td>
      <td>7.4</td>
      <td>25.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WNW</td>
      <td>44.0</td>
      <td>NNW</td>
      <td>...</td>
      <td>44.0</td>
      <td>25.0</td>
      <td>1010.6</td>
      <td>1007.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.2</td>
      <td>24.3</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-12-03</td>
      <td>Albury</td>
      <td>12.9</td>
      <td>25.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WSW</td>
      <td>46.0</td>
      <td>W</td>
      <td>...</td>
      <td>38.0</td>
      <td>30.0</td>
      <td>1007.6</td>
      <td>1008.7</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.2</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-12-04</td>
      <td>Albury</td>
      <td>9.2</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NE</td>
      <td>24.0</td>
      <td>SE</td>
      <td>...</td>
      <td>45.0</td>
      <td>16.0</td>
      <td>1017.6</td>
      <td>1012.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.1</td>
      <td>26.5</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-12-05</td>
      <td>Albury</td>
      <td>17.5</td>
      <td>32.3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>41.0</td>
      <td>ENE</td>
      <td>...</td>
      <td>82.0</td>
      <td>33.0</td>
      <td>1010.8</td>
      <td>1006.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>17.8</td>
      <td>29.7</td>
      <td>No</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-dddda83a-a0ab-446c-92d4-9c5a8a4885af')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-dddda83a-a0ab-446c-92d4-9c5a8a4885af button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-dddda83a-a0ab-446c-92d4-9c5a8a4885af');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  



```python
col_names = df.columns

col_names
```

<pre>
Index(['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'RainToday', 'RainTomorrow'],
      dtype='object')
</pre>

```python
# view summary of dataset

df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 145460 entries, 0 to 145459
Data columns (total 23 columns):
 #   Column         Non-Null Count   Dtype  
---  ------         --------------   -----  
 0   Date           145460 non-null  object 
 1   Location       145460 non-null  object 
 2   MinTemp        143975 non-null  float64
 3   MaxTemp        144199 non-null  float64
 4   Rainfall       142199 non-null  float64
 5   Evaporation    82670 non-null   float64
 6   Sunshine       75625 non-null   float64
 7   WindGustDir    135134 non-null  object 
 8   WindGustSpeed  135197 non-null  float64
 9   WindDir9am     134894 non-null  object 
 10  WindDir3pm     141232 non-null  object 
 11  WindSpeed9am   143693 non-null  float64
 12  WindSpeed3pm   142398 non-null  float64
 13  Humidity9am    142806 non-null  float64
 14  Humidity3pm    140953 non-null  float64
 15  Pressure9am    130395 non-null  float64
 16  Pressure3pm    130432 non-null  float64
 17  Cloud9am       89572 non-null   float64
 18  Cloud3pm       86102 non-null   float64
 19  Temp9am        143693 non-null  float64
 20  Temp3pm        141851 non-null  float64
 21  RainToday      142199 non-null  object 
 22  RainTomorrow   142193 non-null  object 
dtypes: float64(16), object(7)
memory usage: 25.5+ MB
</pre>
**변수의 종류**





이 섹션에서는 데이터 세트를 범주형 변수와 숫자 변수로 분리한다. 데이터 세트에는 범주형 변수와 숫자 변수가 혼합되어 있다. 범주형 변수에는 개체 데이터 유형이 있습니다. 숫자 변수의 데이터 유형은 float64이다.





우선 범주형 변수를 찾자.



```python
# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
```

<pre>
There are 7 categorical variables

The categorical variables are : ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
</pre>

```python
# view the categorical variables

df[categorical].head()
```


  <div id="df-4a17c07c-c635-4cd7-8e9c-f22f03218fac">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Location</th>
      <th>WindGustDir</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>RainToday</th>
      <th>RainTomorrow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008-12-01</td>
      <td>Albury</td>
      <td>W</td>
      <td>W</td>
      <td>WNW</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008-12-02</td>
      <td>Albury</td>
      <td>WNW</td>
      <td>NNW</td>
      <td>WSW</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-12-03</td>
      <td>Albury</td>
      <td>WSW</td>
      <td>W</td>
      <td>WSW</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-12-04</td>
      <td>Albury</td>
      <td>NE</td>
      <td>SE</td>
      <td>E</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-12-05</td>
      <td>Albury</td>
      <td>W</td>
      <td>ENE</td>
      <td>NW</td>
      <td>No</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-4a17c07c-c635-4cd7-8e9c-f22f03218fac')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-4a17c07c-c635-4cd7-8e9c-f22f03218fac button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-4a17c07c-c635-4cd7-8e9c-f22f03218fac');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  


**범주형 변수 요약**





- 날짜 변수가 있습니다. '날짜' 열로 표시된다.





- 6개의 범주형 변수가 있다. 이들은 `Location`, `WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday` 및 `RainTomorrow`로 제공된다.





- `RainToday`와 `RainTomorrow`의 두 가지 이진 범주형 변수가 있다.





- `RainTomorrow`는 타깃 변수이다.



```python
# check missing values in categorical variables

df[categorical].isnull().sum()
```

<pre>
Date                0
Location            0
WindGustDir     10326
WindDir9am      10566
WindDir3pm       4228
RainToday        3261
RainTomorrow     3267
dtype: int64
</pre>

```python
# print categorical variables containing missing values

cat1 = [var for var in categorical if df[var].isnull().sum()!=0]

print(df[cat1].isnull().sum())
```

<pre>
WindGustDir     10326
WindDir9am      10566
WindDir3pm       4228
RainToday        3261
RainTomorrow     3267
dtype: int64
</pre>
누락된 값이 포함된 데이터 세트에는 5개의 범주형 변수만 있음을 알 수 있다. 이들은 `WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday` 및 `RainTomorrow`이다.


**범주형 변수의 빈도수**





이제 범주형 변수의 빈도 수를 확인하자.



```python
# view frequency of categorical variables

for var in categorical: 
    
    print(df[var].value_counts())
```

<pre>
2013-11-12    49
2014-09-01    49
2014-08-23    49
2014-08-24    49
2014-08-25    49
              ..
2007-11-29     1
2007-11-28     1
2007-11-27     1
2007-11-26     1
2008-01-31     1
Name: Date, Length: 3436, dtype: int64
Canberra            3436
Sydney              3344
Darwin              3193
Melbourne           3193
Brisbane            3193
Adelaide            3193
Perth               3193
Hobart              3193
Albany              3040
MountGambier        3040
Ballarat            3040
Townsville          3040
GoldCoast           3040
Cairns              3040
Launceston          3040
AliceSprings        3040
Bendigo             3040
Albury              3040
MountGinini         3040
Wollongong          3040
Newcastle           3039
Tuggeranong         3039
Penrith             3039
Woomera             3009
Nuriootpa           3009
Cobar               3009
CoffsHarbour        3009
Moree               3009
Sale                3009
PerthAirport        3009
PearceRAAF          3009
Witchcliffe         3009
BadgerysCreek       3009
Mildura             3009
NorfolkIsland       3009
MelbourneAirport    3009
Richmond            3009
SydneyAirport       3009
WaggaWagga          3009
Williamtown         3009
Dartmoor            3009
Watsonia            3009
Portland            3009
Walpole             3006
NorahHead           3004
SalmonGums          3001
Katherine           1578
Nhil                1578
Uluru               1578
Name: Location, dtype: int64
W      9915
SE     9418
N      9313
SSE    9216
E      9181
S      9168
WSW    9069
SW     8967
SSW    8736
WNW    8252
NW     8122
ENE    8104
ESE    7372
NE     7133
NNW    6620
NNE    6548
Name: WindGustDir, dtype: int64
N      11758
SE      9287
E       9176
SSE     9112
NW      8749
S       8659
W       8459
SW      8423
NNE     8129
NNW     7980
ENE     7836
NE      7671
ESE     7630
SSW     7587
WNW     7414
WSW     7024
Name: WindDir9am, dtype: int64
SE     10838
W      10110
S       9926
WSW     9518
SSE     9399
SW      9354
N       8890
WNW     8874
NW      8610
ESE     8505
E       8472
NE      8263
SSW     8156
NNW     7870
ENE     7857
NNE     6590
Name: WindDir3pm, dtype: int64
No     110319
Yes     31880
Name: RainToday, dtype: int64
No     110316
Yes     31877
Name: RainTomorrow, dtype: int64
</pre>

```python
# view frequency distribution of categorical variables

for var in categorical: 
    
    print(df[var].value_counts()/np.float(len(df)))
```

<pre>
2013-11-12    0.000337
2014-09-01    0.000337
2014-08-23    0.000337
2014-08-24    0.000337
2014-08-25    0.000337
                ...   
2007-11-29    0.000007
2007-11-28    0.000007
2007-11-27    0.000007
2007-11-26    0.000007
2008-01-31    0.000007
Name: Date, Length: 3436, dtype: float64
Canberra            0.023622
Sydney              0.022989
Darwin              0.021951
Melbourne           0.021951
Brisbane            0.021951
Adelaide            0.021951
Perth               0.021951
Hobart              0.021951
Albany              0.020899
MountGambier        0.020899
Ballarat            0.020899
Townsville          0.020899
GoldCoast           0.020899
Cairns              0.020899
Launceston          0.020899
AliceSprings        0.020899
Bendigo             0.020899
Albury              0.020899
MountGinini         0.020899
Wollongong          0.020899
Newcastle           0.020892
Tuggeranong         0.020892
Penrith             0.020892
Woomera             0.020686
Nuriootpa           0.020686
Cobar               0.020686
CoffsHarbour        0.020686
Moree               0.020686
Sale                0.020686
PerthAirport        0.020686
PearceRAAF          0.020686
Witchcliffe         0.020686
BadgerysCreek       0.020686
Mildura             0.020686
NorfolkIsland       0.020686
MelbourneAirport    0.020686
Richmond            0.020686
SydneyAirport       0.020686
WaggaWagga          0.020686
Williamtown         0.020686
Dartmoor            0.020686
Watsonia            0.020686
Portland            0.020686
Walpole             0.020665
NorahHead           0.020652
SalmonGums          0.020631
Katherine           0.010848
Nhil                0.010848
Uluru               0.010848
Name: Location, dtype: float64
W      0.068163
SE     0.064746
N      0.064024
SSE    0.063358
E      0.063117
S      0.063028
WSW    0.062347
SW     0.061646
SSW    0.060058
WNW    0.056730
NW     0.055837
ENE    0.055713
ESE    0.050681
NE     0.049038
NNW    0.045511
NNE    0.045016
Name: WindGustDir, dtype: float64
N      0.080833
SE     0.063846
E      0.063083
SSE    0.062643
NW     0.060147
S      0.059528
W      0.058153
SW     0.057906
NNE    0.055885
NNW    0.054860
ENE    0.053870
NE     0.052736
ESE    0.052454
SSW    0.052159
WNW    0.050969
WSW    0.048288
Name: WindDir9am, dtype: float64
SE     0.074508
W      0.069504
S      0.068239
WSW    0.065434
SSE    0.064616
SW     0.064306
N      0.061116
WNW    0.061006
NW     0.059192
ESE    0.058470
E      0.058243
NE     0.056806
SSW    0.056070
NNW    0.054104
ENE    0.054015
NNE    0.045305
Name: WindDir3pm, dtype: float64
No     0.758415
Yes    0.219167
Name: RainToday, dtype: float64
No     0.758394
Yes    0.219146
Name: RainTomorrow, dtype: float64
</pre>
***레이블 수: cardinality***





범주형 변수 내의 레이블 수를 **cardinality**라고 한다. 변수 내의 레이블 수가 많은 것을 **high cardinality**라고 한다. 높은 cardinality는는 기계 학습 모델에서 몇 가지 심각한 문제를 일으킬 수 있다. 따라서 high cardinality를 확인하자.



```python
# check for cardinality in categorical variables

for var in categorical:
    
    print(var, ' contains ', len(df[var].unique()), ' labels')
```

<pre>
Date  contains  3436  labels
Location  contains  49  labels
WindGustDir  contains  17  labels
WindDir9am  contains  17  labels
WindDir3pm  contains  17  labels
RainToday  contains  3  labels
RainTomorrow  contains  3  labels
</pre>
전처리가 필요한 `Date` 변수가 있음을 알 수 있다. 다음 섹션에서 전처리를 수행한다.





다른 모든 변수는 상대적으로 적은 수의 변수를 포함한다.


***변수의 기능 엔지니어링***




```python
df['Date'].dtypes
```

<pre>
dtype('O')
</pre>
`Date` 변수의 데이터 타입이 객체임을 알 수 있다. 현재 개체로 코딩된 날짜를 datetime 형식으로 구문 분석하자.



```python
# parse the dates, currently coded as strings, into datetime format

df['Date'] = pd.to_datetime(df['Date'])
```


```python
# extract year from date

df['Year'] = df['Date'].dt.year

df['Year'].head()
```

<pre>
0    2008
1    2008
2    2008
3    2008
4    2008
Name: Year, dtype: int64
</pre>

```python
# extract month from date

df['Month'] = df['Date'].dt.month

df['Month'].head()
```

<pre>
0    12
1    12
2    12
3    12
4    12
Name: Month, dtype: int64
</pre>

```python
# extract day from date

df['Day'] = df['Date'].dt.day

df['Day'].head()
```

<pre>
0    1
1    2
2    3
3    4
4    5
Name: Day, dtype: int64
</pre>

```python
# again view the summary of dataset

df.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 145460 entries, 0 to 145459
Data columns (total 26 columns):
 #   Column         Non-Null Count   Dtype         
---  ------         --------------   -----         
 0   Date           145460 non-null  datetime64[ns]
 1   Location       145460 non-null  object        
 2   MinTemp        143975 non-null  float64       
 3   MaxTemp        144199 non-null  float64       
 4   Rainfall       142199 non-null  float64       
 5   Evaporation    82670 non-null   float64       
 6   Sunshine       75625 non-null   float64       
 7   WindGustDir    135134 non-null  object        
 8   WindGustSpeed  135197 non-null  float64       
 9   WindDir9am     134894 non-null  object        
 10  WindDir3pm     141232 non-null  object        
 11  WindSpeed9am   143693 non-null  float64       
 12  WindSpeed3pm   142398 non-null  float64       
 13  Humidity9am    142806 non-null  float64       
 14  Humidity3pm    140953 non-null  float64       
 15  Pressure9am    130395 non-null  float64       
 16  Pressure3pm    130432 non-null  float64       
 17  Cloud9am       89572 non-null   float64       
 18  Cloud3pm       86102 non-null   float64       
 19  Temp9am        143693 non-null  float64       
 20  Temp3pm        141851 non-null  float64       
 21  RainToday      142199 non-null  object        
 22  RainTomorrow   142193 non-null  object        
 23  Year           145460 non-null  int64         
 24  Month          145460 non-null  int64         
 25  Day            145460 non-null  int64         
dtypes: datetime64[ns](1), float64(16), int64(3), object(6)
memory usage: 28.9+ MB
</pre>
`Date` 변수에서 생성된 세 개의 추가 열이 있음을 알 수 있다. 이제 데이터 세트에서 원래 `Date` 변수를 삭제한다.



```python
# drop the original Date variable

df.drop('Date', axis=1, inplace = True)
```


```python
# preview the dataset again

df.head()
```


  <div id="df-81176461-ea47-45a0-86b8-97055bc53030">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>...</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday</th>
      <th>RainTomorrow</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Albury</td>
      <td>13.4</td>
      <td>22.9</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>44.0</td>
      <td>W</td>
      <td>WNW</td>
      <td>...</td>
      <td>1007.1</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>21.8</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albury</td>
      <td>7.4</td>
      <td>25.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WNW</td>
      <td>44.0</td>
      <td>NNW</td>
      <td>WSW</td>
      <td>...</td>
      <td>1007.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.2</td>
      <td>24.3</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Albury</td>
      <td>12.9</td>
      <td>25.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>WSW</td>
      <td>46.0</td>
      <td>W</td>
      <td>WSW</td>
      <td>...</td>
      <td>1008.7</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.2</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albury</td>
      <td>9.2</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NE</td>
      <td>24.0</td>
      <td>SE</td>
      <td>E</td>
      <td>...</td>
      <td>1012.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.1</td>
      <td>26.5</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Albury</td>
      <td>17.5</td>
      <td>32.3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>41.0</td>
      <td>ENE</td>
      <td>NW</td>
      <td>...</td>
      <td>1006.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>17.8</td>
      <td>29.7</td>
      <td>No</td>
      <td>No</td>
      <td>2008</td>
      <td>12</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-81176461-ea47-45a0-86b8-97055bc53030')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-81176461-ea47-45a0-86b8-97055bc53030 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-81176461-ea47-45a0-86b8-97055bc53030');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  


이제 데이터 세트에서 `Date` 변수가 제거된 것을 볼 수 있다.



***범주형 변수 살펴보기***





이제 범주형 변수를 하나씩 살펴보자.



```python
# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
```

<pre>
There are 6 categorical variables

The categorical variables are : ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
</pre>
데이터 세트에 6개의 범주형 변수가 있음을 알 수 있다. `Date` 변수가 제거 되었다. 먼저 범주형 변수에서 누락된 값을 확인하자.



```python
# check for missing values in categorical variables 

df[categorical].isnull().sum()
```

<pre>
Location            0
WindGustDir     10326
WindDir9am      10566
WindDir3pm       4228
RainToday        3261
RainTomorrow     3267
dtype: int64
</pre>
`WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday`, `RainTomorrow` 변수에 누락된 값이 포함된 것을 볼 수 있다. 이러한 변수를 하나씩 살펴보자\.


**`Location` 변수 살펴보기**



```python
# print number of labels in Location variable

print('Location contains', len(df.Location.unique()), 'labels')
```

<pre>
Location contains 49 labels
</pre>

```python
# check labels in location variable

df.Location.unique()
```

<pre>
array(['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
       'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
       'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
       'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
       'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
       'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
       'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
       'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
       'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
       'AliceSprings', 'Darwin', 'Katherine', 'Uluru'], dtype=object)
</pre>

```python
# check frequency distribution of values in Location variable

df.Location.value_counts()
```

<pre>
Canberra            3436
Sydney              3344
Darwin              3193
Melbourne           3193
Brisbane            3193
Adelaide            3193
Perth               3193
Hobart              3193
Albany              3040
MountGambier        3040
Ballarat            3040
Townsville          3040
GoldCoast           3040
Cairns              3040
Launceston          3040
AliceSprings        3040
Bendigo             3040
Albury              3040
MountGinini         3040
Wollongong          3040
Newcastle           3039
Tuggeranong         3039
Penrith             3039
Woomera             3009
Nuriootpa           3009
Cobar               3009
CoffsHarbour        3009
Moree               3009
Sale                3009
PerthAirport        3009
PearceRAAF          3009
Witchcliffe         3009
BadgerysCreek       3009
Mildura             3009
NorfolkIsland       3009
MelbourneAirport    3009
Richmond            3009
SydneyAirport       3009
WaggaWagga          3009
Williamtown         3009
Dartmoor            3009
Watsonia            3009
Portland            3009
Walpole             3006
NorahHead           3004
SalmonGums          3001
Katherine           1578
Nhil                1578
Uluru               1578
Name: Location, dtype: int64
</pre>

```python
# let's do One Hot Encoding of Location variable
# get k-1 dummy variables after One Hot Encoding 
# preview the dataset with head() method

pd.get_dummies(df.Location, drop_first=True).head()
```


  <div id="df-a582adbe-23c1-4369-a346-a3d94cd4fc96">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Albany</th>
      <th>Albury</th>
      <th>AliceSprings</th>
      <th>BadgerysCreek</th>
      <th>Ballarat</th>
      <th>Bendigo</th>
      <th>Brisbane</th>
      <th>Cairns</th>
      <th>Canberra</th>
      <th>Cobar</th>
      <th>...</th>
      <th>Townsville</th>
      <th>Tuggeranong</th>
      <th>Uluru</th>
      <th>WaggaWagga</th>
      <th>Walpole</th>
      <th>Watsonia</th>
      <th>Williamtown</th>
      <th>Witchcliffe</th>
      <th>Wollongong</th>
      <th>Woomera</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a582adbe-23c1-4369-a346-a3d94cd4fc96')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-a582adbe-23c1-4369-a346-a3d94cd4fc96 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a582adbe-23c1-4369-a346-a3d94cd4fc96');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  


**`WindGustDir` 변수 탐색하기**



```python
# print number of labels in WindGustDir variable

print('WindGustDir contains', len(df['WindGustDir'].unique()), 'labels')
```

<pre>
WindGustDir contains 17 labels
</pre>

```python
# check labels in WindGustDir variable

df['WindGustDir'].unique()
```

<pre>
array(['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', nan, 'ENE',
       'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'], dtype=object)
</pre>

```python
# check frequency distribution of values in WindGustDir variable

df.WindGustDir.value_counts()
```

<pre>
W      9915
SE     9418
N      9313
SSE    9216
E      9181
S      9168
WSW    9069
SW     8967
SSW    8736
WNW    8252
NW     8122
ENE    8104
ESE    7372
NE     7133
NNW    6620
NNE    6548
Name: WindGustDir, dtype: int64
</pre>

```python
# let's do One Hot Encoding of WindGustDir variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()
```


  <div id="df-12fc08aa-4bc7-4b94-8abd-35b4461f8fec">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENE</th>
      <th>ESE</th>
      <th>N</th>
      <th>NE</th>
      <th>NNE</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-12fc08aa-4bc7-4b94-8abd-35b4461f8fec')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-12fc08aa-4bc7-4b94-8abd-35b4461f8fec button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-12fc08aa-4bc7-4b94-8abd-35b4461f8fec');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  



```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).sum(axis=0)
```

<pre>
ENE     8104
ESE     7372
N       9313
NE      7133
NNE     6548
NNW     6620
NW      8122
S       9168
SE      9418
SSE     9216
SSW     8736
SW      8967
W       9915
WNW     8252
WSW     9069
NaN    10326
dtype: int64
</pre>
`WindGustDir` 변수에 10326개의 누락된 값이 있음을 알 수 있다.


**`WindDir9am` 변수 살펴보기**



```python
# print number of labels in WindDir9am variable

print('WindDir9am contains', len(df['WindDir9am'].unique()), 'labels')
```

<pre>
WindDir9am contains 17 labels
</pre>

```python
# check labels in WindDir9am variable

df['WindDir9am'].unique()
```

<pre>
array(['W', 'NNW', 'SE', 'ENE', 'SW', 'SSE', 'S', 'NE', nan, 'SSW', 'N',
       'WSW', 'ESE', 'E', 'NW', 'WNW', 'NNE'], dtype=object)
</pre>

```python
# check frequency distribution of values in WindDir9am variable

df['WindDir9am'].value_counts()
```

<pre>
N      11758
SE      9287
E       9176
SSE     9112
NW      8749
S       8659
W       8459
SW      8423
NNE     8129
NNW     7980
ENE     7836
NE      7671
ESE     7630
SSW     7587
WNW     7414
WSW     7024
Name: WindDir9am, dtype: int64
</pre>

```python
# let's do One Hot Encoding of WindDir9am variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()
```


  <div id="df-66cf1f29-3dc1-4d6c-9abc-32544449d4a1">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENE</th>
      <th>ESE</th>
      <th>N</th>
      <th>NE</th>
      <th>NNE</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-66cf1f29-3dc1-4d6c-9abc-32544449d4a1')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-66cf1f29-3dc1-4d6c-9abc-32544449d4a1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-66cf1f29-3dc1-4d6c-9abc-32544449d4a1');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  



```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).sum(axis=0)
```

<pre>
ENE     7836
ESE     7630
N      11758
NE      7671
NNE     8129
NNW     7980
NW      8749
S       8659
SE      9287
SSE     9112
SSW     7587
SW      8423
W       8459
WNW     7414
WSW     7024
NaN    10566
dtype: int64
</pre>
`WindDir9am` 변수에 10566개의 누락된 값이 있음을 알 수 있다.





**Explore `WindDir3pm` variable**



```python
# print number of labels in WindDir3pm variable

print('WindDir3pm contains', len(df['WindDir3pm'].unique()), 'labels')
```

<pre>
WindDir3pm contains 17 labels
</pre>

```python
# check labels in WindDir3pm variable

df['WindDir3pm'].unique()
```

<pre>
array(['WNW', 'WSW', 'E', 'NW', 'W', 'SSE', 'ESE', 'ENE', 'NNW', 'SSW',
       'SW', 'SE', 'N', 'S', 'NNE', nan, 'NE'], dtype=object)
</pre>

```python
# check frequency distribution of values in WindDir3pm variable

df['WindDir3pm'].value_counts()
```

<pre>
SE     10838
W      10110
S       9926
WSW     9518
SSE     9399
SW      9354
N       8890
WNW     8874
NW      8610
ESE     8505
E       8472
NE      8263
SSW     8156
NNW     7870
ENE     7857
NNE     6590
Name: WindDir3pm, dtype: int64
</pre>

```python
# let's do One Hot Encoding of WindDir3pm variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()
```


  <div id="df-ff3a9a33-cf9b-4846-8059-f5f2f747560d">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ENE</th>
      <th>ESE</th>
      <th>N</th>
      <th>NE</th>
      <th>NNE</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ff3a9a33-cf9b-4846-8059-f5f2f747560d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ff3a9a33-cf9b-4846-8059-f5f2f747560d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ff3a9a33-cf9b-4846-8059-f5f2f747560d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  



```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).sum(axis=0)
```

<pre>
ENE     7857
ESE     8505
N       8890
NE      8263
NNE     6590
NNW     7870
NW      8610
S       9926
SE     10838
SSE     9399
SSW     8156
SW      9354
W      10110
WNW     8874
WSW     9518
NaN     4228
dtype: int64
</pre>
`WindDir3pm` 변수에 4228개의 누락된 값이 있다.


**`RainToday` 변수 살펴보기**



```python
# print number of labels in RainToday variable

print('RainToday contains', len(df['RainToday'].unique()), 'labels')
```

<pre>
RainToday contains 3 labels
</pre>

```python
# check labels in WindGustDir variable

df['RainToday'].unique()
```

<pre>
array(['No', 'Yes', nan], dtype=object)
</pre>

```python
# check frequency distribution of values in WindGustDir variable

df.RainToday.value_counts()
```

<pre>
No     110319
Yes     31880
Name: RainToday, dtype: int64
</pre>

```python
# let's do One Hot Encoding of RainToday variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()
```


  <div id="df-2b0671b9-52ba-49ac-b6e7-7945a376003d">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Yes</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-2b0671b9-52ba-49ac-b6e7-7945a376003d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-2b0671b9-52ba-49ac-b6e7-7945a376003d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-2b0671b9-52ba-49ac-b6e7-7945a376003d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  



```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).sum(axis=0)
```

<pre>
Yes    31880
NaN     3261
dtype: int64
</pre>
`RainToday` 변수에 3261개의 누락된 값이 있다.


**`RainTomorrow` 변수 살펴보기**



```python
# print number of labels in RainToday variable

print('RainTomorrow contains', len(df['RainTomorrow'].unique()), 'labels')
```

<pre>
RainTomorrow contains 3 labels
</pre>

```python
# check labels in WindGustDir variable

df['RainTomorrow'].unique()
```

<pre>
array(['No', 'Yes', nan], dtype=object)
</pre>

```python
# check frequency distribution of values in WindGustDir variable

df.RainTomorrow.value_counts()
```

<pre>
No     110316
Yes     31877
Name: RainTomorrow, dtype: int64
</pre>

```python
# let's do One Hot Encoding of RainToday variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.RainTomorrow, drop_first=True, dummy_na=True).head()
```


  <div id="df-bc356042-07d0-4f90-a045-65bc476c3303">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Yes</th>
      <th>NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-bc356042-07d0-4f90-a045-65bc476c3303')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-bc356042-07d0-4f90-a045-65bc476c3303 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-bc356042-07d0-4f90-a045-65bc476c3303');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  



```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.RainTomorrow, drop_first=True, dummy_na=True).sum(axis=0)
```

<pre>
Yes    31877
NaN     3267
dtype: int64
</pre>
`RainTomorrow` 변수에 3267개의 누락된 값이 있다.


**수치형 변수 살펴보기**



```python
# find numerical variables

numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)
```

<pre>
There are 19 numerical variables

The numerical variables are : ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'Year', 'Month', 'Day']
</pre>

```python
# view the numerical variables

df[numerical].head()
```


  <div id="df-ed0e1bcd-ea17-43ad-983f-4563e9791563">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13.4</td>
      <td>22.9</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>20.0</td>
      <td>24.0</td>
      <td>71.0</td>
      <td>22.0</td>
      <td>1007.7</td>
      <td>1007.1</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>21.8</td>
      <td>2008</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.4</td>
      <td>25.1</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>4.0</td>
      <td>22.0</td>
      <td>44.0</td>
      <td>25.0</td>
      <td>1010.6</td>
      <td>1007.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.2</td>
      <td>24.3</td>
      <td>2008</td>
      <td>12</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.9</td>
      <td>25.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>46.0</td>
      <td>19.0</td>
      <td>26.0</td>
      <td>38.0</td>
      <td>30.0</td>
      <td>1007.6</td>
      <td>1008.7</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.2</td>
      <td>2008</td>
      <td>12</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.2</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24.0</td>
      <td>11.0</td>
      <td>9.0</td>
      <td>45.0</td>
      <td>16.0</td>
      <td>1017.6</td>
      <td>1012.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.1</td>
      <td>26.5</td>
      <td>2008</td>
      <td>12</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.5</td>
      <td>32.3</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.0</td>
      <td>7.0</td>
      <td>20.0</td>
      <td>82.0</td>
      <td>33.0</td>
      <td>1010.8</td>
      <td>1006.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>17.8</td>
      <td>29.7</td>
      <td>2008</td>
      <td>12</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ed0e1bcd-ea17-43ad-983f-4563e9791563')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ed0e1bcd-ea17-43ad-983f-4563e9791563 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ed0e1bcd-ea17-43ad-983f-4563e9791563');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  


**수치형 변수 요약**





- 16개의 수치 변수가 있습니다.





- `MinTemp`, `MaxTemp`, `Rainfall`, `Evaporation`, `Sunshine`, `WindGustSpeed`, `WindSpeed9am`, `WindSpeed3pm`, `Humidity9am`, `Humidity3pm`, `Pressure9am`, ` Pressure3pm`, `Cloud9am`, `Cloud3pm`, `Temp9am` 및 `Temp3pm`.





- 수치형 변수는 모두 연속형이다.


**수치형 변수 내에서 문제 탐색**





이제 수치형 변수를 살펴보자.





**수치형 변수의 누락된 값**



```python
# check missing values in numerical variables

df[numerical].isnull().sum()
```

<pre>
MinTemp           1485
MaxTemp           1261
Rainfall          3261
Evaporation      62790
Sunshine         69835
WindGustSpeed    10263
WindSpeed9am      1767
WindSpeed3pm      3062
Humidity9am       2654
Humidity3pm       4507
Pressure9am      15065
Pressure3pm      15028
Cloud9am         55888
Cloud3pm         59358
Temp9am           1767
Temp3pm           3609
Year                 0
Month                0
Day                  0
dtype: int64
</pre>
16개의 모든 수치 변수에 누락된 값이 포함되어 있음을 알 수 있다.


**수치형 변수의 이상치**



```python
# view summary statistics in numerical variables

print(round(df[numerical].describe()),2)
```

<pre>
        MinTemp   MaxTemp  Rainfall  Evaporation  Sunshine  WindGustSpeed  \
count  143975.0  144199.0  142199.0      82670.0   75625.0       135197.0   
mean       12.0      23.0       2.0          5.0       8.0           40.0   
std         6.0       7.0       8.0          4.0       4.0           14.0   
min        -8.0      -5.0       0.0          0.0       0.0            6.0   
25%         8.0      18.0       0.0          3.0       5.0           31.0   
50%        12.0      23.0       0.0          5.0       8.0           39.0   
75%        17.0      28.0       1.0          7.0      11.0           48.0   
max        34.0      48.0     371.0        145.0      14.0          135.0   

       WindSpeed9am  WindSpeed3pm  Humidity9am  Humidity3pm  Pressure9am  \
count      143693.0      142398.0     142806.0     140953.0     130395.0   
mean           14.0          19.0         69.0         52.0       1018.0   
std             9.0           9.0         19.0         21.0          7.0   
min             0.0           0.0          0.0          0.0        980.0   
25%             7.0          13.0         57.0         37.0       1013.0   
50%            13.0          19.0         70.0         52.0       1018.0   
75%            19.0          24.0         83.0         66.0       1022.0   
max           130.0          87.0        100.0        100.0       1041.0   

       Pressure3pm  Cloud9am  Cloud3pm   Temp9am   Temp3pm      Year  \
count     130432.0   89572.0   86102.0  143693.0  141851.0  145460.0   
mean        1015.0       4.0       5.0      17.0      22.0    2013.0   
std            7.0       3.0       3.0       6.0       7.0       3.0   
min          977.0       0.0       0.0      -7.0      -5.0    2007.0   
25%         1010.0       1.0       2.0      12.0      17.0    2011.0   
50%         1015.0       5.0       5.0      17.0      21.0    2013.0   
75%         1020.0       7.0       7.0      22.0      26.0    2015.0   
max         1040.0       9.0       9.0      40.0      47.0    2017.0   

          Month       Day  
count  145460.0  145460.0  
mean        6.0      16.0  
std         3.0       9.0  
min         1.0       1.0  
25%         3.0       8.0  
50%         6.0      16.0  
75%         9.0      23.0  
max        12.0      31.0   2
</pre>
자세히 살펴보면 `Rainfall`, `Evaporation`, `WindSpeed9am` 및 `WindSpeed3pm` 열에 이상값이 포함될 수 있음을 알 수 있다.





위의 변수에서 이상값을 시각화하기 위해 boxplots 그릴 것이다.



```python
# draw boxplots to visualize outliers

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')


plt.subplot(2, 2, 2)
fig = df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')


plt.subplot(2, 2, 3)
fig = df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')


plt.subplot(2, 2, 4)
fig = df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')
```

<pre>
Text(0, 0.5, 'WindSpeed3pm')
</pre>
<img src="data:image/png;base64,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kiVL9Mcff+jee+/Vb7/9pp9++kmzZ8+WdCXgS01NtVo0wdfXVy1bttSmTZuuG6zl5ORYreickZEhScrNzVVubq5degfgnFq1aqU6derIz89Pv//+u9U4r06dOmrWrJnOnj2rVq1acb0AyrmSXAOKHaxFRETY0ssNxcTEKDY2Vnv37pWnp6dat26t6dOnq169epaa7OxsjRw5Up988olycnLUrVs3LVy4UP7+/paao0ePavDgwfrhhx9UsWJFDRw4UDExMXJ15YY8ACXz0EMPadq0afrjjz/Uq1cvjRgxQvv371fdunW1du1affPNN5Y6ACiLJk+erBdffPGaYC0rK0uTJ0+2W7D2yiuvKCMjQ/Xr15eLi4tMJpPeeOMNRUVFSfq/uSwLj/kKtm80z2VMTIwmT558zf61a9de850AlD+PP/64ZsyYoebNm6tr167y8PBQTk6Ofv31V6WkpGjMmDH67rvvHN0mAAfLysoqdm2xk6dXX33VpmZuJDExUUOHDtUDDzygvLw8jR8/Xl27dtXu3bvl7e0tSRoxYoS++eYbff755/L19dWwYcMUGRmpn3/+WdKViW979eqlgIAAbdy4USdPntTTTz8tNzc3TZs2ze49AyjbOnTooLvuukunTp3S+vXrLUGaJHl6ekqSqlevzhLsAMoss9lc5FMKv/32m/z8/Oz2OZ999pmWLVum5cuX67777tP27ds1fPhwBQUFaeDAgTafd9y4cYqOjrZsZ2RkKDg4WF27dpWPj489WgfgxHr27Kn7779fY8aM0bvvvmvZX6dOHX3yySd65JFHHNgdgNKi4I734nDoLV0Fz7AX+OCDD1S9enWlpKSoXbt2On/+vN5//30tX77ccnfI0qVL1aBBAyUnJ6tVq1Zau3atdu/ere+//17+/v5q2rSpXn/9dY0dO1avvfaa3N3dHfHVADgpFxcXLV68WP369bvmWME/NBctWiQXF5c73RoA3FZVqlSRwWCQwWDQvffeaxWumUwmXbx4US+++KLdPm/06NF65ZVXLI90hoWF6ciRI4qJidHAgQMVEBAgSUpLS1NgYKDlfWlpaWratOl1z+vh4SEPD49r9ru5ucnNzc1u/QNwXq6urkX+gODq6sp1AoAklehaYFOwZjKZNGfOHH322Wc6evSoLl++bHX8zJkztpxW58+flyTLr6EpKSnKzc21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ANXWzp079dlnn2nlypVq1qxZiVqQJycDAIDqxOkr1vbu3avWrVvL29tb+/bt09dff23/t3v3breGKygo0IQJExQXF6fAwEA1atRIU6ZMcXhCi81m08SJExUTE6PAwED16tVLP/zwg1tzAKheBgwYoA8++EB79+5V165d9T//8z/q2rWr9u3bpw8++ID1NwBUazVq1NCAAQOUmJioWrVqKTw83OEfAHiqUaNG2V+3a9fOYV/x7eLzAOBqvGwe/BzhadOm6eWXX9bbb7+tZs2a6csvv9SDDz6o559/3v700RdffFGpqal6++23FRcXpwkTJmjv3r367rvvFBAQUKavw2PYAZSmoKBAGzdu1OrVq3Xbbbepe/fuXKkGwI76oXLg9wSgSPEnF9tsNlmtVq1atUp9+/aVn59fif0Aqi9n6genbwWtSF988YX69++v22+/XZLUsGFDLVmyRDt37pR06WQ3e/ZsPfvss+rfv78k6Z133lFUVJSWL1+ugQMHGpYdQOXn4+OjxMREZWdnKzExkaYaABRz9uxZHTx4UJLUpEkT1a5d2+BEAFA2/v7+pY77+PiooKCggtMAqOxcaqx9+eWX+u9//6vjx4/LYrE47HPnuhodO3bUG2+8oe+//17XX3+9vvnmG23dutW+jtuRI0d05swZ9erVy/6e8PBwtW/fXtu2bbtsYy0vL095eXn2bbPZLEmyWq2yWq1uyw+g8is6J3BuAPBH1fW8YDabNXz4cL333nv2P0B9fHx0zz33aO7cudwOCsDj/fFv2CI01QC4wunG2nvvvaf7779fffr00bp169S7d299//33ysjI0B133OHWcE8//bTMZrPi4+Ptnx48//zzuu+++yRJZ86ckSRFRUU5vC8qKsq+rzSpqamaPHlyifF169YpKCjIjd8BgKpi/fr1RkcA4GFycnKMjmCIhx9+WF9//bVWrlypDh06SLr0NL1Ro0Zp6NCheu+99wxOCACle/zxxzVnzhxJl/5mPHv2rH1f8atui5YdAoCycHqNtZYtW2ro0KEaPny4QkND9c033yguLk5Dhw5VTExMqQ0rV7333nsaO3asZsyYoWbNmmn37t0aPXq0Xn75ZQ0ZMkRffPGFOnXqpFOnTikmJsb+vrvvvlteXl56//33Sz1uaVesxcbG6pdffmHtDQAOrFar1q9fr1tuuaXEk+8AVG9ms1m1atWqdmt3BQcHa+3atercubPDeFpamm699VZlZ2cblKx0rLEGoLji66hdDuurASjXNdYOHz5sX/PM399f2dnZ8vLy0pgxY9SjRw+3NtbGjh2rp59+2n5LZ4sWLXTs2DGlpqZqyJAhio6OliRlZGQ4NNYyMjLUqlWryx7XZDLJZDKVGPfz8+MPZwCl4vwA4I+q6zkhMjKy1Ns9w8PDVbNmTQMSAQAAGMfb2TfUrFlTFy5ckCTVrVtX+/btkySdP3/e7bdE5OTkyNvbMaKPj48KCwslSXFxcYqOjtaGDRvs+81ms3bs2GG/NQEAAADu8+yzzyolJcVh2Y0zZ85o7NixmjBhgoHJAODK+vTpY39dr149h33Ft4vPA4CrcfqKta5du2r9+vVq0aKF7rrrLo0aNUqfffaZ1q9fr549e7o13F/+8hc9//zzql+/vpo1a6avv/5aL7/8sv72t79JunQZ7+jRozV16lRdd911iouL04QJE1SnTh0lJSW5NQsAAACk1157TYcOHVL9+vVVv359SdLx48dlMpn0888/6/XXX7fP/eqrr4yKCQAlrFu3zv76xIkTslqtWrVqlfr27Ss/Pz/7baLF5wHA1TjdWHv11VeVm5srSfrHP/4hPz8/ffHFF0pOTtazzz7r1nCvvPKKJkyYoGHDhuns2bOqU6eOhg4dqokTJ9rnjBs3TtnZ2XrkkUd0/vx5de7cWWvWrFFAQIBbswAAAEB8eAkAAFCM0w8vqIpY1BbA5fzxk0wAKEL9UDnwewJQpPiDC2w222WvWCvaD6D6cqZ+cHqNtezsbG3ZskXvv/++li5dqvT0dE46AAAA1VRGRoaOHz9udAwAuKrevXvbXw8fPtxhX/Ht4vMA4GrK3FgrLCzUuHHjdM0116h79+669957dc8996ht27aKi4vTihUryjMnAAAADHThwgUNGjRIDRo00JAhQ2SxWDR8+HDFxMQoLi5OiYmJMpvNRscEgMtau3at/fW8efPk7++vpKQk+fv7a968eaXOA4CrKXNj7ZlnntHKlSv13//+V2vXrlXnzp31wgsv6LvvvtP999+vu+66i0UeAVQpWVlZSk5O1qhRo5ScnKysrCyjIwGAYZ555hmlp6frySef1PHjx3X33Xdry5YtSktL08aNG/XLL7/oxRdfNDomAFzR1e624m4sAM4q8xprderU0fvvv68uXbpIkn766SfFx8frl19+kclk0pQpU7R69Wp98cUX5Rq4PLD2BoA/ateunXbt2lVivG3bttq5c6cBiQB4mupWP9SvX19vv/22unfvrlOnTqlevXr65JNP1K9fP0nSp59+qieeeEIHDhwwOKmj6vZ7AnBlxddRuxyaawDKZY21rKws1a1b174dExOj3NxcnTt3TpKUnJysb775xsXIAOA5ippqXl5eGjRokGbNmqVBgwbJy8tLu3btUrt27YyOCAAV7uzZs2rcuLGkSx+4BgYG6vrrr7fvb968uU6cOGFUPAC4quJNtcDAQFksFi1fvlwWi0WBgYGlzgOAqylzY61FixZasmSJffu///2vQkJCFB0dLenSGmwmk8n9CQGgAmVlZdmbajk5OXrzzTcVFxenN998Uzk5OfbmGreFAqhuIiMj9fPPP9u3+/fvrxo1ati3s7KyqAUBVBo5OTlX3AaAsipzY+25557TlClT1L59eyUmJmrw4MGaNGmSff+aNWvUunXrcgkJABVl8ODBkqRBgwYpICDAYV9AQIDuvfdeh3kAUF20bNnS4Rb5xYsXq3bt2vbtXbt2KSEhwYhoAAAAhilzY61nz57asWOHevXqpbZt22rVqlUaPXq0ff+TTz6pDRs2lEdGAKgwhw8flnTpnFZQUKDNmzdry5Yt2rx5swoKCpSSkuIwDwCqi3fffVf33HPPZfdHRUXp+eefr8BEAAAAxvN1ZvINN9ygG264obyyAIDhGjVqpL179+rxxx/XsWPHdPToUUnSyy+/rIYNGyo2NtY+DwCqk4iIiCvuv+222yooCQD8eayjBsBdytRY27NnT5kP2LJlS5fDAIDRFi1apNDQUG3evFm33367Fi1apJMnT6pevXp64YUX9Omnn9rnAUB1QS0IoCqw2Ww8FRSA25WpsdaqVSt5eXnZTzBXOhkVFBS4JxkAGCAwMFD+/v6yWCz69NNPFR4erjZt2mjFihX2pprJZHJ4chQAVHXFa8Gr/VFKLQgAAKqTMq2xduTIEf344486cuSIPvroI8XFxWnevHn6+uuv9fXXX2vevHlq1KiRPvzww/LOCwDlKi0tTRaLxb4A9+LFi/XEE09o8eLFkqSEhATl5eUpLS3NyJgAUKGK14IffvghtSCASqmst39ymygAZ5TpirUGDRrYX991112aM2eO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lXmui6dPAFVEda8fKkstWN1/TwAub/exX5X02nYtf+xmtWrAFbgAfudM/VDmK9b+KDg4WBEREYqIiFDNmjXl6+ur/fv3u3o4APA4NptNFotFy5cvl8VioZEGAMVQCwIAADixxlphYaG+/PJLbdq0SRs3btTnn3+u7Oxs1a1bV927d9fcuXPVvXv38swKABXKmavXAKCqoxYEAAAoqcyNtRo1aig7O1vR0dHq3r27Zs2apW7duqlRo0blmQ8AAAAegFoQAACgpDI31mbMmKHu3bvr+uuvL888AAAA8EDUggAAACWVubE2dOjQ8swBAAAAD0YtCAAAUJLLDy8wwgsvvCAvLy+NHj3aPpabm6vhw4crMjJSISEhSk5OVkZGhnEhAQAAAAAAUC1Umsbarl279Prrr6tly5YO42PGjNGKFSu0dOlSbd68WadOndKAAQMMSgkAAAAAAIDqosy3ghopKytL9913nxYsWKCpU6faxzMzM/Wf//xHixcvVo8ePSRJCxcuVEJCgrZv366bb7651OPl5eUpLy/Pvm02myVJVqtVVqu1HL8TAFUJ5wugeuMcAAAAgErRWBs+fLhuv/129erVy6Gxlp6eLqvVql69etnH4uPjVb9+fW3btu2yjbXU1FRNnjy5xPi6desUFBTk/m8AQJW0atUqoyMAMFBOTo7REQAAAGAwj2+svffee/rqq6+0a9euEvvOnDkjf39/1ahRw2E8KipKZ86cuewxx48fr5SUFPu22WxWbGysevfurbCwMLdlB1C19e3b1+gIAAxUdMU7AAAAqi+PbqydOHFCo0aN0vr16xUQEOC245pMJplMphLjfn5+8vPzc9vXAVC1cb4AqjfOAQAAAPDohxekp6fr7NmzuvHGG+Xr6ytfX19t3rxZc+bMka+vr6KiomSxWHT+/HmH92VkZCg6OtqY0AAAAAAAAKgWPLqx1rNnT+3du1e7d++2/7vpppt033332V/7+flpw4YN9vccPHhQx48fV4cOHQxMDqAqsNlsslgsWr58uSwWi2w2m9GRAAAAAAAexKNvBQ0NDVXz5s0dxoKDgxUZGWkff+ihh5SSkqKIiAiFhYVp5MiR6tChw2UfXAAAZeXl5SWLxeKwDQAAAABAEY9urJXFrFmz5O3treTkZOXl5alPnz6aN2+e0bEAVFI2m82hgebv73/ZeQAAAACA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JvvYpk2b1L9/f8XFxWn79u0O4927dy9xi8Bbb72lGjVqaO3atUpISFBISIhuvfVWnT592v7egoICpaSkqEaNGoqMjNS4ceNks9ns+8+fP6+0tDS9+OKL6t69uxo0aKB27dpp/Pjx+utf/2qf5+Xlpddee0233XabAgMDde211+qDDz5w+J5OnDihu+++WzVq1FBERIT69++vo0ePOsz597//rYSEBAUEBCg+Pl7z5s1z2L9z5061bt1aAQEBuummm/T111877H///ffVsmVLTZw4UY0bN1ZiYqKmT5+uuXPn6sKFC5Iufdr5P//zP6pbt66CgoLUokULLVmyxOE43bp108iRIzV69GjVrFlTUVFRWrBggbKzs/Xggw8qNDRUjRs31urVq6/ymwQAAHBEnedanVezZk099thjuummm9SgQQP17NlTw4YNU1pamn3OP//5T7Vq1Uqvv/66YmNjFRQUpLvvvluZmZn2OUW3w06bNk1RUVGqUaOGnnvuOeXn52vs2LGKiIhQvXr1tHDhwqv8JgG4G401AG7VvXt3bdy40b69ceNGdevWTYmJifbxixcvaseOHerevXupx8jJydFLL72kRYsWacuWLTp+/LiefPJJ+/6ZM2fqrbfe0ptvvqmtW7fqt99+07Jly+z7Q0JCFBISouXLlysvL++KeSdMmKDk5GR98803uu+++zRw4EDt379fkmS1WtWnTx+FhoYqLS1Nn3/+ub0AtFgskqR3331XEydO1PPPP6/9+/dr2rRpmjBhgt5++21Jl26R6Nevn5o2bar09HT985//dPheJCkvL8/+CW+RwMBA5ebmKj09XZKUm5urNm3a6NNPP9W+ffv0yCOPaPDgwdq5c6fD+95++23VqlVLO3fu1MiRI/XYY4/prrvuUseOHfXVV1+pd+/eGjx4sHJycq74cwEAAPgj6jzn67w/OnXqlD766CMlJiY6jB86dEj//e9/tWLFCq1Zs0Zff/21hg0b5jDns88+06lTp7Rlyxa9/PLLmjRpkvr166eaNWtqx44devTRRzV06FCdPHnyihkAuJkNANxowYIFtuDgYJvVarWZzWabr6+v7ezZs7bFixfbunbtarPZbLYNGzbYJNmOHTtm27hxo02S7dy5czabzWZbuHChTZLt0KFD9mPOnTvXFhUVZd+OiYmxTZ8+3b5ttVpt9erVs/Xv398+9sEHH9hq1qxpCwgIsHXs2NE2fvx42zfffOOQVZLt0UcfdRhr37697bHHHrPZbDbbokWLbE2aNLEVFhba9+fl5dkCAwNta9eutdlsNlujRo1sixcvdjjGlClTbB06dLDZbDbb66+/bouMjLRdvHjRvv+1116zSbJ9/fXXNpvNZlu7dq3N29vbtnjxYlt+fr7t5MmTti5dutgklTh2cbfffrvtiSeesG8nJibaOnfubN/Oz8+3BQcH2wYPHmwfO336tE2Sbdu2bZc9LgAAQGmo85yv84oMHDjQFhgYaJNk+8tf/uLwnkmTJtl8fHxsJ0+etI+tXr3a5u3tbTt9+rTNZrPZhgwZYmvQoIGtoKDAPqdJkya2Ll262LeLar8lS5bYAFQcrlgD4FbdunVTdna2du3apbS0NF1//fW65pprlJiYaF9/Y9OmTbr22mtVv379Uo8RFBSkRo0a2bdjYmJ09uxZSVJmZqZOnz6t9u3b2/f7+vrqpptucjhGcnKyTp06pU8++US33nqrNm3apBtvvFFvvfWWw7wOHTqU2C76JPObb77RoUOHFBoaav90NCIiQrm5uTp8+LCys7N1+PBhPfTQQ/b9ISEhmjp1qg4fPixJ2r9/v1q2bOlwRdofv2bv3r01Y8YMPfroozKZTLr++uvVt29fSZfW5ZAu3RYxZcoUtWjRQhEREQoJCdHatWt1/Phxh2O1bNnS/trHx0eRkZFq0aKFfSwqKkqS7D9PAACAsqLOc77OKzJr1ix99dVX+vjjj3X48GGlpKQ47K9fv77q1q3rcJzCwkIdPHjQPtasWTN7bShdquuK13lFtR91HlCxeHgBALdq3Lix6tWrp40bN+rcuXP2y9zr1Kmj2NhYffHFF9q4caN69Ohx2WP4+fk5bHt5eTmsrVFWAQEBuuWWW3TLLbdowoQJ+vvf/65JkybpgQceKNP7s7Ky1KZNG7377rsl9l1zzTXKysqSJC1YsMChAJQuFTbOSElJ0ZgxY3T69GnVrFlTR48e1fjx43XttddKkmbMmKF//etfmj17tlq0aKHg4GCNHj3afqtCkdJ+dsXHvLy8JEmFhYVO5QMAAKDOu8TZOk+SoqOjFR0drfj4eEVERKhLly6aMGGCYmJiynyMq9V5RWPUeUDF4oo1AG5XtFjtpk2bHB6/3rVrV61evVo7d+687LobVxMeHq6YmBjt2LHDPpafn29fi+xKmjZtquzsbIex4gvtFm0nJCRIkm688Ub98MMPql27tho3buzwLzw8XFFRUapTp45+/PHHEvvj4uIkSQkJCdqzZ49yc3Mv+zWLeHl5qU6dOgoMDNSSJUsUGxurG2+8UZL0+eefq3///ho0aJBuuOEGXXvttfr+++/L8BMDAABwH+o81+q84ooaX8XXiDt+/LhOnTrlcBxvb+9ye7IqAPehsQbA7bp3766tW7dq9+7dDguzJiYm6vXXX5fFYnG54JKkUaNG6YUXXtDy5ct14MABDRs2zP60KenSEzR79Oih//3f/9WePXt05MgRLV26VNOnT1f//v0djrV06VK9+eab+v777zVp0iTt3LlTI0aMkCTdd999qlWrlvr376+0tDQdOXJEmzZt0uOPP25fFHby5MlKTU3VnDlz9P3332vv3r1auHChXn75ZUnSvffeKy8vLz388MP67rvvtGrVKr300kslvqcZM2Zo7969+vbbbzVlyhS98MILmjNnjv0T0euuu07r16/XF198of3792vo0KHKyMhw+WcIAADgCuo85+q8VatWaeHChdq3b5+OHj2qTz/9VI8++qg6deqkhg0b2ucFBARoyJAh+uabb5SWlqbHH39cd999t6Kjo13+WQKoGNwKCsDtunfvrosXLyo+Pt6+ppd0qeC6cOGC/XHtrnriiSd0+vRpDRkyRN7e3vrb3/6mO+64w/5I8pCQELVv316zZs3S4cOHZbVaFRsbq4cffljPPPOMw7EmT56s9957T8OGDVNMTIyWLFmipk2bSrq0BsiWLVv01FNPacCAAbpw4YLq1q2rnj17KiwsTJL097//XUFBQZoxY4bGjh2r4OBgtWjRQqNHj7ZnWbFihR599FG1bt1aTZs21Ysvvqjk5GSHHKtXr9bzzz+vvLw83XDDDfr4449122232fc/++yz+vHHH9WnTx8FBQXpkUceUVJSksNj2AEAAMobdZ5zdV5gYKAWLFigMWPGKC8vT7GxsRowYICefvpph6yNGzfWgAED1LdvX/3222/q16+f5s2b5/LPEUDF8bK5ckM7AFQBXl5eWrZsmZKSkoyOAgAAADeqTHXeP//5Ty1fvly7d+82OgoAF3ArKAAAAAAAAOACGmsAAAAAAACAC7gVFAAAAAAAAHABV6wBAAAAAAAALqCxBgAAAAAAALiAxhoAAAAAAADgAhprAAAAAAAAgAtorAEAAAAAAAAuoLEGAAAAAAAAuIDGGgAAAAAAAOACGmsAAAAAAACAC/4fTTbg/kRUZ6kAAAAASUVORK5CYII="/>

위의 상자 그림은 이러한 변수에 많은 이상치가 있음을 확인한다.


**변수 분포 확인**





이제 히스토그램을 플로팅하여 분포가 정상인지 왜곡되었는지 확인한다. 변수가 정규 분포를 따른다면 `Exterme Value Analysis`을 하고, 왜곡되어 있으면 IQR(Interquantile range)을 찾는다.



```python
# plot histogram to check distribution

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')
```

<pre>
Text(0, 0.5, 'RainTomorrow')
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABPAAAANBCAYAAABu8kBdAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAADpYUlEQVR4nOzde1yUdf7//ycgB9EADwmSqGy6ng+FhVjbWiKjUWm5bhZrpK5+MjCRVstSPFWmJYpJslZqfdM87KZb2hITplbiCWXz3MmyLQfbPJCaMML1+2N+XDp5BAcY4HG/3bitc12vec/7/ZTNNy9mrsvDMAxDAAAAAAAAANySZ1VPAAAAAAAAAMCl0cADAAAAAAAA3BgNPAAAAAAAAMCN0cADAAAAAAAA3BgNPAAAAAAAAMCN0cADAAAAAAAA3BgNPAAAAAAAAMCN0cADAAAAAAAA3Fidqp5AbVJSUqIff/xR1113nTw8PKp6OgAAoBowDEO//PKLQkND5enJ717dFfs8AABQVmXZ59HAq0Q//vijwsLCqnoaAACgGvr+++/VrFmzqp4GLoF9HgAAKK+r2efRwKtE1113nSTHX0xAQIDLx7fb7crKylJMTIy8vb1dPn51UNszqO3rl8hAIgOJDGr7+qWalUFBQYHCwsLMfQTcE/u8ykEO55CFAzk4kIMDOZxDFg7unkNZ9nk08CpR6ccpAgICKmxj5+/vr4CAALf8xqwMtT2D2r5+iQwkMpDIoLavX6qZGfCxTPfGPq9ykMM5ZOFADg7k4EAO55CFQ3XJ4Wr2eVxIBQAAAAAAAHBjNPAAAAAAAAAAN0YDDwAAAAAAAHBjNPAAAAAAAAAAN0YDDwAAAAAAAHBjNPAAAAAAAAAAN0YDDwAAAAAAAHBjNPAAAAAAAAAAN0YDDwAAAAAAAHBjNPAAAAAAAAAAN0YDDwAAAAAAAHBjVdrA27hxo+69916FhobKw8NDq1evvmTtY489Jg8PD82ZM8fp+NGjRxUXF6eAgAAFBQVp2LBhOnnypFPN559/rj/84Q/y8/NTWFiYZs6cecH4K1euVNu2beXn56dOnTrpgw8+cDpvGIZSUlLUtGlT1a1bV9HR0fryyy/LvXYAAIDarLi4WBMnTlR4eLjq1q2rG2+8UdOmTZNhGGbN1ey/KmsvCAAAUJWqtIF36tQpdenSRenp6ZetW7VqlTZv3qzQ0NALzsXFxWnPnj2yWq1as2aNNm7cqBEjRpjnCwoKFBMToxYtWig3N1cvvfSSJk+erAULFpg1mzZt0kMPPaRhw4Zp586d6t+/v/r376/du3ebNTNnztTcuXOVkZGhLVu2qF69erJYLDpz5owLkgAAAKhdZsyYofnz52vevHnat2+fZsyYoZkzZ+qVV14xa65m/1VZe0EAAICqVKcqX7xv377q27fvZWt++OEHjRo1Sh9++KFiY2Odzu3bt0+ZmZnatm2bunXrJkl65ZVXdPfdd+vll19WaGiolixZoqKiIi1cuFA+Pj7q0KGD8vLylJqaam7u0tLS1KdPH40dO1aSNG3aNFmtVs2bN08ZGRkyDENz5szRhAkT1K9fP0nSW2+9peDgYK1evVqDBg1ydTTXpOPkD1VY7FHV0yiXb1+MvXIRAACo9jZt2qR+/fqZ+7uWLVvqnXfe0datWyXpqvZflbUXdCfs8wAAqJ2qtIF3JSUlJRo8eLDGjh2rDh06XHA+JydHQUFB5oZNkqKjo+Xp6aktW7bo/vvvV05Oju644w75+PiYNRaLRTNmzNCxY8fUoEED5eTkKDk52Wlsi8VifqT34MGDstlsio6ONs8HBgYqMjJSOTk5l2zgFRYWqrCw0HxcUFAgSbLb7bLb7WUP5ApKx/T1NK5Q6b6uNZfS51dEvtVBbV+/RAYSGUhkUNvXL9WsDGrCGi6mR48eWrBggb744gv9/ve/13/+8x99+umnSk1NlXR1+6/K2gsCAABUNbdu4M2YMUN16tTRE088cdHzNptNTZo0cTpWp04dNWzYUDabzawJDw93qgkODjbPNWjQQDabzTx2fs35Y5z/vIvVXMz06dM1ZcqUC45nZWXJ39//ks+7VtO6lVTY2BXNVdebsVqtLhmnuqrt65fIQCIDiQxq+/qlmpHB6dOnq3oKFeLpp59WQUGB2rZtKy8vLxUXF+v5559XXFycpKvbf1XWXvBi+EVt2bkil5rUnL9WZOFADg7k4EAO55CFg7vnUJZ5uW0DLzc3V2lpadqxY4c8PKrnxwTGjx/v9NvcgoIChYWFKSYmRgEBAS5/PbvdLqvVqonbPVVYUj0z2z3Zck3PL82gd+/e8vb2dtGsqo/avn6JDCQykMigtq9fqlkZlDaGapoVK1ZoyZIlWrp0qfmx1qSkJIWGhio+Pr6qp3dF/KK27Fx5Y5Ca0Jx3FbJwIAcHcnAgh3PIwsFdcyjLL2rdtoH3ySef6MiRI2revLl5rLi4WE8++aTmzJmjb7/9ViEhITpy5IjT886ePaujR48qJCREkhQSEqL8/HynmtLHV6o5/3zpsaZNmzrVdO3a9ZJr8PX1la+v7wXHvb29K/SHicISj2p7bRRX5VLRGbu72r5+iQwkMpDIoLavX6oZGVT3+V/K2LFj9fTTT5uXIunUqZO+++47TZ8+XfHx8Ve1/6qsveDF8IvasrvWX9RKNas5f63IwoEcHMjBgRzOIQsHd8+hLL+oddsG3uDBg52ueSI5rkUyePBgDRkyRJIUFRWl48ePKzc3VxEREZKkdevWqaSkRJGRkWbNs88+K7vdbv5lWa1WtWnTRg0aNDBrsrOzlZSUZL6W1WpVVFSUJCk8PFwhISHKzs42N4wFBQXasmWLRo4cWWEZAAAA1FSnT5+Wp6en0zEvLy+VlDjeYXY1+6/K2gteDL+oLTtX5lITmvOuQhYO5OBADg7kcA5ZOLhrDmWZU5U28E6ePKmvvvrKfHzw4EHl5eWpYcOGat68uRo1auRU7+3trZCQELVp00aS1K5dO/Xp00fDhw9XRkaG7Ha7EhMTNWjQIIWGhkqSHn74YU2ZMkXDhg3TU089pd27dystLU2zZ882xx09erT++Mc/atasWYqNjdWyZcu0fft2LViwQJLk4eGhpKQkPffcc2rdurXCw8M1ceJEhYaGqn///hWcEgAAQM1z77336vnnn1fz5s3VoUMH7dy5U6mpqRo6dKikq9t/VdZeEAAAoKpVaQNv+/btuvPOO83HpR9DiI+P1+LFi69qjCVLligxMVG9evWSp6enBgwYoLlz55rnAwMDlZWVpYSEBEVERKhx48ZKSUnRiBEjzJoePXpo6dKlmjBhgp555hm1bt1aq1evVseOHc2acePG6dSpUxoxYoSOHz+u22+/XZmZmfLz87vGFAAAAGqfV155RRMnTtTjjz+uI0eOKDQ0VP/3f/+nlJQUs+Zq9l+VtRcEAACoSlXawOvZs6cM4+rvpPXtt99ecKxhw4ZaunTpZZ/XuXNnffLJJ5etGThwoAYOHHjJ8x4eHpo6daqmTp16VXMFAADApV133XWaM2eO5syZc8maq9l/VdZeEAAAoCp5XrkEAAAAAAAAQFWhgQcAAAAAAAC4MRp4AAAAAAAAgBujgQcAAAAAAAC4MRp4AAAAAAAAgBujgQcAAAAAAAC4MRp4AAAAAAAAgBujgQcAAAAAAAC4MRp4AAAAAAAAgBujgQcAAAAAAAC4MRp4AAAAAAAAgBujgQcAAAAAAAC4MRp4AAAAAAAAgBujgQcAAAAAAAC4MRp4AAAAAAAAgBujgQcAAAAAAAC4MRp4AAAAAAAAgBujgQcAAAAAAAC4MRp4AAAAAAAAgBujgQcAAAAAAAC4MRp4AAAAAAAAgBujgQcAAAAAAAC4MRp4AAAAAAAAgBujgQcAAAAAAAC4MRp4AAAAAAAAgBujgQcAAAAAAAC4MRp4AAAAAAAAgBujgQcAAAAAAAC4MRp4AAAAAAAAgBujgQcAAAAAAAC4MRp4AAAAAAAAgBujgQcAAIAq0bJlS3l4eFzwlZCQIEk6c+aMEhIS1KhRI9WvX18DBgxQfn6+0xiHDh1SbGys/P391aRJE40dO1Znz551qlm/fr1uvvlm+fr6qlWrVlq8ePEFc0lPT1fLli3l5+enyMhIbd26tcLWDQAAUFY08AAAAFAltm3bpsOHD5tfVqtVkjRw4EBJ0pgxY/T+++9r5cqV2rBhg3788Uc98MAD5vOLi4sVGxuroqIibdq0SW+++aYWL16slJQUs+bgwYOKjY3VnXfeqby8PCUlJemvf/2rPvzwQ7Nm+fLlSk5O1qRJk7Rjxw516dJFFotFR44cqaQkAAAALo8GHgAAAKrE9ddfr5CQEPNrzZo1uvHGG/XHP/5RJ06c0BtvvKHU1FTdddddioiI0KJFi7Rp0yZt3rxZkpSVlaW9e/fq7bffVteuXdW3b19NmzZN6enpKioqkiRlZGQoPDxcs2bNUrt27ZSYmKg//elPmj17tjmP1NRUDR8+XEOGDFH79u2VkZEhf39/LVy4sEpyAQAA+K06VT0BAAAAoKioSG+//baSk5Pl4eGh3Nxc2e12RUdHmzVt27ZV8+bNlZOTo+7duysnJ0edOnVScHCwWWOxWDRy5Ejt2bNHN910k3JycpzGKK1JSkoyXzc3N1fjx483z3t6eio6Olo5OTmXnG9hYaEKCwvNxwUFBZIku90uu91+TVlcTOmYvp6Gy8euLK7IpXSMisi4uiELB3JwIAcHcjiHLBzcPYeyzIsGHgAAAKrc6tWrdfz4cT366KOSJJvNJh8fHwUFBTnVBQcHy2azmTXnN+9Kz5eeu1xNQUGBfv31Vx07dkzFxcUXrdm/f/8l5zt9+nRNmTLlguNZWVny9/e/8oLLaVq3kgobu6J98MEHLhur9OPWIItS5OBADg7kcA5ZOLhrDqdPn77q2ipt4G3cuFEvvfSScnNzdfjwYa1atUr9+/eX5OhCTpgwQR988IG++eYbBQYGKjo6Wi+++KJCQ0PNMY4ePapRo0bp/fffl6enpwYMGKC0tDTVr1/frPn888+VkJCgbdu26frrr9eoUaM0btw4p7msXLlSEydO1LfffqvWrVtrxowZuvvuu83zhmFo0qRJeu2113T8+HHddtttmj9/vlq3bl2xIQEAANQCb7zxhvr27eu0z3Nn48ePV3Jysvm4oKBAYWFhiomJUUBAgMtfz263y2q1auJ2TxWWeLh8/Mqwe7LlmscozaF3797y9vZ2wayqL7JwIAcHcnAgh3PIwsHdcyh9B//VqNIG3qlTp9SlSxcNHTrU6YLEkqMLuWPHDk2cOFFdunTRsWPHNHr0aN13333avn27WRcXF2de9Nhut2vIkCEaMWKEli5dKskRRkxMjKKjo5WRkaFdu3Zp6NChCgoK0ogRIyRJmzZt0kMPPaTp06frnnvu0dKlS9W/f3/t2LFDHTt2lCTNnDlTc+fO1Ztvvqnw8HBNnDhRFotFe/fulZ+fXyUlBgAAUPN89913+uijj/Tuu++ax0JCQlRUVKTjx487vQsvPz9fISEhZs1v7xZbepfa82t+e+fa/Px8BQQEqG7duvLy8pKXl9dFa0rHuBhfX1/5+vpecNzb27tCf0AoLPFQYXH1bOC5MpeKzrk6IQsHcnAgBwdyOIcsHNw1h7LMqUpvYtG3b18999xzuv/++y84FxgYKKvVqj//+c9q06aNunfvrnnz5ik3N1eHDh2SJO3bt0+ZmZl6/fXXFRkZqdtvv12vvPKKli1bph9//FGStGTJEhUVFWnhwoXq0KGDBg0apCeeeEKpqanma6WlpalPnz4aO3as2rVrp2nTpunmm2/WvHnzJDnefTdnzhxNmDBB/fr1U+fOnfXWW2/pxx9/1OrVqys+KAAAgBps0aJFatKkiWJjY81jERER8vb2VnZ2tnnswIEDOnTokKKioiRJUVFR2rVrl9PdYq1WqwICAtS+fXuz5vwxSmtKx/Dx8VFERIRTTUlJibKzs80aAACAqlat7kJ74sQJeXh4mL+FzcnJUVBQkLp162bWREdHy9PTU1u2bDFr7rjjDvn4+Jg1FotFBw4c0LFjx8yai13cuPTCxQcPHpTNZnOqCQwMVGRk5GUvbgwAAIDLKykp0aJFixQfH686dc59OCQwMFDDhg1TcnKyPv74Y+Xm5mrIkCGKiopS9+7dJUkxMTFq3769Bg8erP/85z/68MMPNWHCBCUkJJjvjnvsscf0zTffaNy4cdq/f79effVVrVixQmPGjDFfKzk5Wa+99prefPNN7du3TyNHjtSpU6c0ZMiQyg0DAADgEqrNTSzOnDmjp556Sg899JB5XRGbzaYmTZo41dWpU0cNGzZ0unBxeHi4U835Fzdu0KDBJS9ufP4Y5z/vYjUXw93Jyu5ac3H3O8xUtNq+fokMJDKQyKC2r1+qWRnUhDVczkcffaRDhw5p6NChF5ybPXu2eY3jwsJCWSwWvfrqq+Z5Ly8vrVmzRiNHjlRUVJTq1aun+Ph4TZ061awJDw/X2rVrNWbMGKWlpalZs2Z6/fXXZbGcux7bgw8+qJ9++kkpKSmy2Wzq2rWrMjMzL9j7AQAAVJVq0cCz2+3685//LMMwNH/+/KqezlXj7mRl56q7k7nrHWYqS21fv0QGEhlIZFDb1y/VjAzKcney6igmJkaGcfFfPvr5+Sk9PV3p6emXfH6LFi2uuH/o2bOndu7cedmaxMREJSYmXnnCAAAAVcDtG3ilzbvvvvtO69atc7qrV0hIiNM1TyTp7NmzOnr06BUvXFx67nI1558vPda0aVOnmq5du15y7tydrOyu9e5k7n6HmYpW29cvkYFEBhIZ1Pb1SzUrg7LcnQwAAAA1k1s38Eqbd19++aU+/vhjNWrUyOl8VFSUjh8/rtzcXEVEREiS1q1bp5KSEkVGRpo1zz77rOx2u7mBt1qtatOmjRo0aGDWZGdnKykpyRz7/Isbh4eHKyQkRNnZ2WbDrqCgQFu2bNHIkSMvOX/uTlZ2rsrFXe8wU1lq+/olMpDIQCKD2r5+qWZkUN3nDwAAgGtXpTexOHnypPLy8pSXlyfJcbOIvLw8HTp0SHa7XX/605+0fft2LVmyRMXFxbLZbLLZbCoqKpIktWvXTn369NHw4cO1detWffbZZ0pMTNSgQYMUGhoqSXr44Yfl4+OjYcOGac+ePVq+fLnS0tKc3hk3evRoZWZmatasWdq/f78mT56s7du3mx+j8PDwUFJSkp577jm999572rVrlx555BGFhoaqf//+lZoZAAAAAAAAapcqfQfe9u3bdeedd5qPS5tq8fHxmjx5st577z1JuuBjqh9//LF69uwpSVqyZIkSExPVq1cv8yLHc+fONWsDAwOVlZWlhIQERUREqHHjxkpJSdGIESPMmh49emjp0qWaMGGCnnnmGbVu3VqrV69Wx44dzZpx48bp1KlTGjFihI4fP67bb79dmZmZ8vPzc3UsAAAAAAAAgKlKG3g9e/a85EWLJV32XKmGDRtq6dKll63p3LmzPvnkk8vWDBw4UAMHDrzkeQ8PD02dOtXprmYAAAAAAABARavSj9ACAAAAAAAAuDwaeAAAAAAAAIAbo4EHAAAAAAAAuDEaeAAAAAAAAIAbo4EHAAAAAAAAuDEaeAAAAAAAAIAbo4EHAAAAAAAAuDEaeAAAAAAAAIAbo4EHAAAAAAAAuDEaeAAAAAAAAIAbo4EHAAAAAAAAuDEaeAAAAAAAAIAbo4EHAAAAAAAAuDEaeAAAAAAAAIAbo4EHAAAAAAAAuDEaeAAAAAAAAIAbo4EHAAAAAAAAuDEaeAAAAAAAAIAbo4EHAAAAAAAAuDEaeAAAAAAAAIAbo4EHAAAAAAAAuDEaeAAAAAAAAIAbo4EHAAAAAAAAuDEaeAAAAAAAAIAbo4EHAAAAAAAAuDEaeAAAAKgSP/zwg/7yl7+oUaNGqlu3rjp16qTt27eb5w3DUEpKipo2baq6desqOjpaX375pdMYR48eVVxcnAICAhQUFKRhw4bp5MmTTjWff/65/vCHP8jPz09hYWGaOXPmBXNZuXKl2rZtKz8/P3Xq1EkffPBBxSwaAACgHGjgAQAAoNIdO3ZMt912m7y9vfXvf/9be/fu1axZs9SgQQOzZubMmZo7d64yMjK0ZcsW1atXTxaLRWfOnDFr4uLitGfPHlmtVq1Zs0YbN27UiBEjzPMFBQWKiYlRixYtlJubq5deekmTJ0/WggULzJpNmzbpoYce0rBhw7Rz5071799f/fv31+7duysnDAAAgCuoU9UTAAAAQO0zY8YMhYWFadGiReax8PBw88+GYWjOnDmaMGGC+vXrJ0l66623FBwcrNWrV2vQoEHat2+fMjMztW3bNnXr1k2S9Morr+juu+/Wyy+/rNDQUC1ZskRFRUVauHChfHx81KFDB+Xl5Sk1NdVs9KWlpalPnz4aO3asJGnatGmyWq2aN2+eMjIyKisSAACAS6KBBwAAgEr33nvvyWKxaODAgdqwYYNuuOEGPf744xo+fLgk6eDBg7LZbIqOjjafExgYqMjISOXk5GjQoEHKyclRUFCQ2byTpOjoaHl6emrLli26//77lZOTozvuuEM+Pj5mjcVi0YwZM3Ts2DE1aNBAOTk5Sk5OdpqfxWLR6tWrLzn/wsJCFRYWmo8LCgokSXa7XXa7/ZqyuZjSMX09DZePXVlckUvpGBWRcXVDFg7k4EAODuRwDlk4uHsOZZkXDTwAAABUum+++Ubz589XcnKynnnmGW3btk1PPPGEfHx8FB8fL5vNJkkKDg52el5wcLB5zmazqUmTJk7n69Spo4YNGzrVnP/OvvPHtNlsatCggWw222Vf52KmT5+uKVOmXHA8KytL/v7+VxNBuUzrVlJhY1c0V15X0Gq1umys6o4sHMjBgRwcyOEcsnBw1xxOnz591bU08AAAAFDpSkpK1K1bN73wwguSpJtuukm7d+9WRkaG4uPjq3h2VzZ+/Hind+0VFBQoLCxMMTExCggIcPnr2e12Wa1WTdzuqcISD5ePXxl2T7Zc8xilOfTu3Vve3t4umFX1RRYO5OBADg7kcA5ZOLh7DqXv4L8aNPAAAABQ6Zo2bar27ds7HWvXrp3++c9/SpJCQkIkSfn5+WratKlZk5+fr65du5o1R44ccRrj7NmzOnr0qPn8kJAQ5efnO9WUPr5STen5i/H19ZWvr+8Fx729vSv0B4TCEg8VFlfPBp4rc6nonKsTsnAgBwdycCCHc8jCwV1zKMucuAstAAAAKt1tt92mAwcOOB374osv1KJFC0mOG1qEhIQoOzvbPF9QUKAtW7YoKipKkhQVFaXjx48rNzfXrFm3bp1KSkoUGRlp1mzcuNHpGjNWq1Vt2rQx73gbFRXl9DqlNaWvAwAAUNVo4AEAAKDSjRkzRps3b9YLL7ygr776SkuXLtWCBQuUkJAgSfLw8FBSUpKee+45vffee9q1a5ceeeQRhYaGqn///pIc79jr06ePhg8frq1bt+qzzz5TYmKiBg0apNDQUEnSww8/LB8fHw0bNkx79uzR8uXLlZaW5vTx19GjRyszM1OzZs3S/v37NXnyZG3fvl2JiYmVngsAAMDF8BFaAAAAVLpbbrlFq1at0vjx4zV16lSFh4drzpw5iouLM2vGjRunU6dOacSIETp+/Lhuv/12ZWZmys/Pz6xZsmSJEhMT1atXL3l6emrAgAGaO3eueT4wMFBZWVlKSEhQRESEGjdurJSUFI0YMcKs6dGjh5YuXaoJEybomWeeUevWrbV69Wp17NixcsIAAAC4Ahp4AAAAqBL33HOP7rnnnkue9/Dw0NSpUzV16tRL1jRs2FBLly697Ot07txZn3zyyWVrBg4cqIEDB15+wgAAAFWEj9ACAAAAAAAAbqxKG3gbN27Uvffeq9DQUHl4eGj16tVO5w3DUEpKipo2baq6desqOjpaX375pVPN0aNHFRcXp4CAAAUFBWnYsGE6efKkU83nn3+uP/zhD/Lz81NYWJhmzpx5wVxWrlyptm3bys/PT506ddIHH3xQ5rkAAAAAAAAArlalDbxTp06pS5cuSk9Pv+j5mTNnau7cucrIyNCWLVtUr149WSwWnTlzxqyJi4vTnj17ZLVatWbNGm3cuNHpmiYFBQWKiYlRixYtlJubq5deekmTJ0/WggULzJpNmzbpoYce0rBhw7Rz5071799f/fv31+7du8s0FwAAAAAAAMDVqvQaeH379lXfvn0ves4wDM2ZM0cTJkxQv379JElvvfWWgoODtXr1ag0aNEj79u1TZmamtm3bpm7dukmSXnnlFd199916+eWXFRoaqiVLlqioqEgLFy6Uj4+POnTooLy8PKWmppqNvrS0NPXp00djx46VJE2bNk1Wq1Xz5s1TRkbGVc0FAAAAAAAAqAhuexOLgwcPymazKTo62jwWGBioyMhI5eTkaNCgQcrJyVFQUJDZvJOk6OhoeXp6asuWLbr//vuVk5OjO+64Qz4+PmaNxWLRjBkzdOzYMTVo0EA5OTlKTk52en2LxWJ+pPdq5nIxhYWFKiwsNB8XFBRIkux2u+x2e/nDuYTSMX09DZePXVmuNZfS51dEvtVBbV+/RAYSGUhkUNvXL9WsDGrCGgAAAHBt3LaBZ7PZJEnBwcFOx4ODg81zNptNTZo0cTpfp04dNWzY0KkmPDz8gjFKzzVo0EA2m+2Kr3OluVzM9OnTNWXKlAuOZ2Vlyd/f/5LPu1bTupVU2NgV7bfXHiwvq9XqknGqq9q+fokMJDKQyKC2r1+qGRmcPn26qqcAAACAKua2DbyaYPz48U7v7CsoKFBYWJhiYmIUEBDg8tez2+2yWq2auN1ThSUeLh+/MuyebLmm55dm0Lt3b3l7e7toVtVHbV+/RAYSGUhkUNvXL9WsDErfwQ8AAIDay20beCEhIZKk/Px8NW3a1Dyen5+vrl27mjVHjhxxet7Zs2d19OhR8/khISHKz893qil9fKWa889faS4X4+vrK19f3wuOe3t7V+gPE4UlHiosrp4NPFflUtEZu7vavn6JDCQykMigtq9fqhkZVPf5AwAA4NpV6V1oLyc8PFwhISHKzs42jxUUFGjLli2KioqSJEVFRen48ePKzc01a9atW6eSkhJFRkaaNRs3bnS6fozValWbNm3UoEEDs+b81ymtKX2dq5kLAAAAAAAAUBGqtIF38uRJ5eXlKS8vT5LjZhF5eXk6dOiQPDw8lJSUpOeee07vvfeedu3apUceeUShoaHq37+/JKldu3bq06ePhg8frq1bt+qzzz5TYmKiBg0apNDQUEnSww8/LB8fHw0bNkx79uzR8uXLlZaW5vTR1tGjRyszM1OzZs3S/v37NXnyZG3fvl2JiYmSdFVzAQAAAAAAACpClX6Edvv27brzzjvNx6VNtfj4eC1evFjjxo3TqVOnNGLECB0/fly33367MjMz5efnZz5nyZIlSkxMVK9eveTp6akBAwZo7ty55vnAwEBlZWUpISFBERERaty4sVJSUjRixAizpkePHlq6dKkmTJigZ555Rq1bt9bq1avVsWNHs+Zq5gIAAAAAAAC4WpU28Hr27CnDMC553sPDQ1OnTtXUqVMvWdOwYUMtXbr0sq/TuXNnffLJJ5etGThwoAYOHHhNcwEAAAAAAABczW2vgQcAAAAAAACABh4AAAAAAADg1mjgAQAAAAAAAG6MBh4AAAAAAADgxmjgAQAAAAAAAG6MBh4AAAAAAADgxmjgAQAAAAAAAG6MBh4AAAAAAADgxmjgAQAAAAAAAG6sXA28hQsX6uDBg66eCwAAAKoB9oIAAACVq1wNvOnTp6tVq1Zq3ry5Bg8erNdff11fffWVq+cGAAAAN8ReEAAAoHKVq4H35Zdf6tChQ5o+fbr8/f318ssvq02bNmrWrJn+8pe/uHqOAAAAcCPsBQEAACpXua+Bd8MNNyguLk6zZ89WWlqaBg8erPz8fC1btsyV8wMAAIAbYi8IAABQeeqU50lZWVlav3691q9fr507d6pdu3b64x//qH/84x+64447XD1HAAAAuBH2ggAAAJWrXA28Pn366Prrr9eTTz6pDz74QEFBQS6eFgAAANwVe0EAAIDKVa6P0Kampuq2227TzJkz1aFDBz388MNasGCBvvjiC1fPDwAAAG6GvSAAAEDlKlcDLykpSe+++67+97//KTMzUz169FBmZqY6duyoZs2auXqOAAAAcCPsBQEAACpXuT5CK0mGYWjnzp1av369Pv74Y3366acqKSnR9ddf78r5AQAAwA2xFwQAAKg85Wrg3Xvvvfrss89UUFCgLl26qGfPnho+fLjuuOMOroECAABQw7EXBAAAqFzl+ght27Zt9dZbb+nnn39Wbm6uZs2apfvuu48NGwAAQC3gqr3g5MmT5eHh4fTVtm1b8/yZM2eUkJCgRo0aqX79+howYIDy8/Odxjh06JBiY2Pl7++vJk2aaOzYsTp79qxTzfr163XzzTfL19dXrVq10uLFiy+YS3p6ulq2bCk/Pz9FRkZq69atZVoLAABARSrXO/BeeuklV88DAAAA1YQr94IdOnTQRx99ZD6uU+fc9nTMmDFau3atVq5cqcDAQCUmJuqBBx7QZ599JkkqLi5WbGysQkJCtGnTJh0+fFiPPPKIvL299cILL0iSDh48qNjYWD322GNasmSJsrOz9de//lVNmzaVxWKRJC1fvlzJycnKyMhQZGSk5syZI4vFogMHDqhJkyYuWysAAEB5lesdeJK0YcMG3XvvvWrVqpVatWql++67T5988okr5wYAAAA35aq9YJ06dRQSEmJ+NW7cWJJ04sQJvfHGG0pNTdVdd92liIgILVq0SJs2bdLmzZslSVlZWdq7d6/efvttde3aVX379tW0adOUnp6uoqIiSVJGRobCw8M1a9YstWvXTomJifrTn/6k2bNnm3NITU3V8OHDNWTIELVv314ZGRny9/fXwoULXZAUAADAtSvXO/DefvttDRkyRA888ICeeOIJSdJnn32mXr16afHixXr44YddOkkAAAC4D1fuBb/88kuFhobKz89PUVFRmj59upo3b67c3FzZ7XZFR0ebtW3btlXz5s2Vk5Oj7t27KycnR506dVJwcLBZY7FYNHLkSO3Zs0c33XSTcnJynMYorUlKSpIkFRUVKTc3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QNUo800sJOm1117TyJEj1bhxY4WEhDhdD8XDw4NNGwAAQA3GXhAAAKBylauB99xzz+n555/XU0895er5AAAAwM2xFwQAAKhcZboGXqljx45p4MCBrp4LAAAAqgH2ggAAAJWrXA28gQMHKisry9VzAQAAQDXAXhAAAKBylesjtK1atdLEiRO1efNmderUSd7e3k7nn3jiCZdMDgAAAO6HvSAAAEDlKtc78BYsWKD69etrw4YNmjdvnmbPnm1+zZkzx6UT/OGHH/SXv/xFjRo1Ut26ddWpUydt377dPG8YhlJSUtS0aVPVrVtX0dHR+vLLL53GOHr0qOLi4hQQEKCgoCANGzZMJ0+edKr5/PPP9Yc//EF+fn4KCwvTzJkzL5jLypUr1bZtW/n5+alTp0764IMPXLpWAACA6qAy94IAAAAo5zvwDh486Op5XNSxY8d022236c4779S///1vXX/99fryyy/VoEEDs2bmzJmaO3eu3nzzTYWHh2vixImyWCzau3ev/Pz8JElxcXE6fPiwrFar7Ha7hgwZohEjRmjp0qWSpIKCAsXExCg6OloZGRnatWuXhg4dqqCgII0YMUKStGnTJj300EOaPn267rnnHi1dulT9+/fXjh071LFjx0rJAwAAwB1U1l4QAAAADuVq4FWWGTNmKCwsTIsWLTKPhYeHm382DENz5szRhAkT1K9fP0nSW2+9peDgYK1evVqDBg3Svn37lJmZqW3btqlbt26SpFdeeUV33323Xn75ZYWGhmrJkiUqKirSwoUL5ePjow4dOigvL0+pqalmAy8tLU19+vTR2LFjJUnTpk2T1WrVvHnzlJGRUVmRAAAAAAAAoJa56gZecnKypk2bpnr16ik5Ofmytampqdc8MUl67733ZLFYNHDgQG3YsEE33HCDHn/8cQ0fPlyS47e/NptN0dHR5nMCAwMVGRmpnJwcDRo0SDk5OQoKCjKbd5IUHR0tT09PbdmyRffff79ycnJ0xx13yMfHx6yxWCyaMWOGjh07pgYNGignJ+eCdVssFq1evdolawUAAHBnVbEXBAAAgMNVN/B27twpu91u/vlSPDw8rn1W/79vvvlG8+fPV3Jysp555hlt27ZNTzzxhHx8fBQfHy+bzSZJCg4OdnpecHCwec5ms6lJkyZO5+vUqaOGDRs61Zz/zr7zx7TZbGrQoIFsNttl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nXr88cc1cuRIHThwQJJ0+PBhdejQQU8++aQOHz6sv/3tb5Kk6667TosXL9bevXuVlpam1157TbNnz67YBQMAAFRTcXFxev/9982GmCR9+OGHOn36tO6//35J0qlTp5ScnKzt27crOztbnp6euv/++1VSUuI01tixY/Xkk09q586dioqK0r333quff/5ZkvTDDz/o7rvv1i233KL//Oc/mj9/vt544w0999xzTmO8+eab8vHx0WeffaaMjAxJkqenp+bOnas9e/bozTff1Lp16zRu3DhJUo8ePTRnzhwFBATo8OHDTvvC85WUlKhfv346evSoNmzYIKvVqm+++eaC5uXXX3+t1atXa82aNVqzZo02bNigF1988RpTBlDhDADAFcXHxxteXl5GvXr1DF9fX0OS4enpafzjH/+45HNiY2ONJ5980nz8xz/+0Rg9erT5uEWLFsZf/vIX83FJSYnRpEkTY/78+eaxLl26GJMmTbrs3F566SUjIiLCfDxp0iSjS5cuTnPv16/flRcJAABQTZ2/Vzv/6/nnnzfsdrvRuHFj46233jLrH3roIePBBx+85Hg//fSTIcnYtWuXYRiGcfDgQUOS8eKLL5o1drvdaNasmTFjxgzDMAzjmWeeMdq0aWOUlJSYNenp6Ub9+vWN4uJiwzAc+8GbbrrpiutZuXKl0ahRI/PxokWLjMDAwAvqWrRoYcyePdswDMPIysoyvLy8jEOHDpnn9+zZY0gytm7dahiGY5/o7+9vFBQUmDVjx441IiMjrzgnAFWLa+ABwFW68847NX/+fJ06dUqzZ89WnTp1NGDAAElScXGxXnjhBa1YsUI//PCDioqKVFhYKH9//8uO2blzZ/PPHh4eCgkJ0ZEjRy77nOXLl2vu3Ln6+uuvdfLkSZ09e1YBAQHXvkAAAIBqrHSvdr6GDRuqTp06+vOf/6wlS5Zo8ODBOnXqlP71r39p2bJlZt2XX36plJQUbdmyRf/73//Md94dOnRIHTt2NOuioqLMP9epU0fdunXTvn37JEn79u1TVFSUPDw8zJrbbrtNJ0+e1H//+181b95ckhQREXHB3D/66CNNnz5d+/fvV0FBgc6ePaszZ87o9OnTV9xPltq3b5/CwsIUFhZmHmvfvr2CgoK0b98+3XLLLZIcH7u97rrrzJqmTZtecf8JoOrxEVoAuEr16tVTq1at1KVLFy1cuFBbtmzRG2+8IUl66aWXlJaWpqeeekoff/yx8vLyZLFYVFRUdNkxvb29nR57eHhc8FGN8+Xk5CguLk5333231qxZo507d+rZZ5+94usAAADUdKV7tfO/Sm8iFhcXp+zsbB05ckSrV69W3bp1ne5Qe++99+ro0aN67bXXtGXLFm3ZskWSKmSPVa9ePafH3377re655x517txZ//znP5Wbm6v09PQKe/2y7j8BuAcaeABQDp6ennrmmWc0YcIE/frrr/rss8/Ur18//eUvf1GXLl30u9/9Tl988YXLX3fTpk1q0aKFnn32WXXr1k2tW7fWd9995/LXAQAAqEl69OihsLAwLV++XEuWLNHAgQPNRtbPP/+sAwcOaMKECerVq5fatWunY8eOXXSczZs3m38+e/ascnNz1a5dO0lSu3btlJOTI8MwzJrPPvtM1113nZo1a3bJueXm5qqkpESzZs1S9+7d9fvf/14//vijU42Pj4+Ki4svu8Z27drp+++/1/fff28e27t3r44fP6727dtf9rkA3B8NPAAop4EDB8rLy0vp6elq3bq1rFarNm3apH379un//u//nO745SqtW7fWoUOHtGzZMn399deaO3euVq1a5fLXAQAAqG4KCwtls9mcvv73v/+Z5x9++GFlZGTIarUqLi7OPN6gQQM1atRICxYs0FdffaV169YpOTn5oq+Rnp6uVatWaf/+/UpISNCxY8c0dOhQSdLjjz+u77//XqNGjdL+/fv1r3/9S5MmTVJycrI8PS/9o3erVq1kt9v1yiuv6JtvvtH/+3//z7y5RamWLVvq5MmT+v/au/O4Kuv8//9PQFYVUEuQRKVsVHKHCU+LuSAno9LRcbIxY8xsNLCAGS0axa3SLFFLijaX+aSTOt9ySg094ZaJG0rj3mbDzCjYjAuucITr90c/rvGECyqHc5DH/XbzNp7rep339b5eFzSv8/I61zsnJ0f/+c9/dObMmUrjxMbGqkOHDhoyZIh27NihrVu36vHHH9d9992n6Ojoq8olAPdDAw8ArlG9evWUlJSk6dOn6w9/+IO6du0qq9WqHj16KDQ0VP3796/2Yz788MNKSUlRUlKSOnfurE2bNmn8+PHVfhwAAIDaJjs7W82aNXP4c88995j7hwwZor179+qWW27R3XffbW739PTUhx9+qLy8PLVv314p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UkhIiMP7QkJCzH2FhYVq2rSpw/569eqpcePGDjEX3tl34ZiFhYVq1KiRCgsLL3uci5k6daomTZpUafvq1asVEBBQlRRckynR5U4b+0bijDzdiM9FtNlsrp5CrUCeqoY8XRk5qpq6lKczZ864egoAAABwMbdu4JWXlys6Olovv/yyJKlLly7avXu3srKylJCQ4OLZXVlaWprDXXvFxcUKDw9XXFycAgMDq/14drtdNptN47d7qqTco9rHv1H4ehqaEl3ulDztnmit1vFcqeLnqU+fPvL29nb1dNwWeaoa8nRl5Khq6mKeKu7gBwAAQN3l1g28Zs2aKTIy0mFbu3bt9P/+3/+TJIWGhkqSioqK1KxZMzOmqKhInTt3NmOOHDniMMb58+d19OhR8/2hoaEqKipyiKl4faWYiv0X4+vrK19f30rbvb29nfqho6TcQyVlNPCuxBl5uhE/TDr75/VGQZ6qhjxdGTmqmrqUp7pyngAAALg0t16F9u6779aBAwcctn399ddq2bKlpJ8WtAgNDVVOTo65v7i4WFu2bJHFYpEkWSwWHT9+XHl5eWbMmjVrVF5erpiYGDNmw4YNDs+YsdlsatOmjbnircVicThORUzFcQAAAAAAAABncOsGXkpKijZv3qyXX35Z3377rRYtWqR33nlHiYmJkiQPDw8lJyfrxRdf1CeffKJdu3bp8ccfV1hYmPr37y/ppzv27r//fo0YMUJbt27Vl19+qaSkJA0ePFhhYWGSpN/+9rfy8fHR8OHDtWfPHi1evFizZ892+Prrs88+q+zsbM2YMUP79+/XxIkTtX37diUlJdV4XgAAAAAAAFB3uPVXaH/5y1/q448/VlpamiZPnqyIiAjNmjVLQ4YMMWPGjh2r06dP66mnntLx48d1zz33KDs7W35+fmbMwoULlZSUpN69e8vT01MDBw7U66+/bu4PCgrS6tWrlZiYqKioKN10001KT0/XU089ZcbcddddWrRokcaNG6cXXnhBt99+u5YtW6b27dvXTDIAAAAAAABQJ7l1A0+SHnzwQT344IOX3O/h4aHJkydr8uTJl4xp3LixFi1adNnjdOzYUV988cVlYwYNGqRBgwZdfsIAAAAAAABANXLrr9ACAAAAAAAAdR0NPAAAALjExIkT5eHh4fCnbdu25v5z584pMTFRTZo0UYMGDTRw4EAVFRU5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69bJw8NDx48fv65xfve736l///7VMidn++GHH+Th4aH8/HxXTwUAAMBpqPMA1AU08ADUOllZWWrYsKHOnz9vbjt16pS8vb3Vo0cPh9iKgq5Zs2Y6fPiwgoKCqnUuP/74o0aNGqUWLVrI19dXoaGhslqt+vLLL6v1ONUlJydHd911lxo2bKjQ0FA999xzDnkEAABwJeq8a/Pf//5X999/v8LCwuTr66vw8HAlJSWpuLjY1VMDUE3quXoCAHC1evbsqVOnTmn79u3q1q2bJOmLL75QaGiotmzZonPnzsnPz0+StHbtWrVo0UJt2rRxylwGDhyo0tJSLViwQLfeequKioqUk5Oj//73v0453vX46quv9MADD+hPf/qT/vznP+vf//63Ro4cqbKyMr322muunh4AAAB13jXy9PRUv3799OKLL+rmm2/Wt99+q8TERB09elSLFi1y9fQAVAPuwANQ67Rp00bNmjXTunXrzG3r1q1Tv379FBERoc2bNzts79mzZ6WvVsyfP1/BwcFatWqV2rVrpwYNGuj+++/X4cOHzfeWlZUpNTVVwcHBatKkicaOHSvDMMz9x48f1xdffKFXXnlFPXv2VMuWLXXnnXcqLS1NDz/8sBnn4eGht956S3379pW/v79uvfVW/fWvf3U4p3/+85/6zW9+o+DgYDVu3Fj9+vXTDz/84BDz3nvvqV27dvLz81Pbtm315ptvOuzfunWrunTpIj8/P0VHR2vnzp0O+xcvXqyOHTsqPT1drVu31n333afp06crMzNTJ0+elPTTv94++uijuuWWWxQQEKAOHTroL3/5i8M4PXr00OjRo5WcnKxGjRopJCRE7777rk6fPq1hw4apYcOGat26tT777LMrXEkAAABH1HnXVuc1atRIo0aNUnR0tFq2bKnevXvr6aef1hdffGHGTJw4UZ07d9bbb7+t8PBwBQQE6De/+Y1OnDhhxlR8jfjll19WSEiIgoODNXnyZJ0/f15jxoxR48aN1bx5c82bN+8KVxJAdaOBB6BW6tmzp9auXWu+Xrt2rXr06KH77rvP3H727Flt2bJFPXv2vOgYZ86c0Wuvvab/+7//04YNG1RQUKA//vGP5v4ZM2Zo/vz5mjt3rjZu3KijR4/q448/Nvc3aNBADRo00LJly1RSUnLZ+Y4fP14DBw7UV199pSFDhmjw4MHat2+fJMlut8tqtaphw4b64osv9OWXX5qFZmlpqSRp4cKFSk9P10svvaR9+/bp5Zdf1vjx47VgwQJJP3215MEHH1RkZKTy8vI0ceJEh3ORpJKSEvNfrCv4+/vr3LlzysvLkySdO3dOUVFRWrFihXbv3q2nnnpKQ4cO1datWx3et2DBAt10003aunWrRo8erVGjRmnQoEG66667tGPHDsXFxWno0KE6c+bMZfMCAADwc9R5V1/n/dyhQ4f00Ucf6b777nPY/u2332rJkiX69NNPlZ2drZ07d+rpp592iFmzZo0OHTqkDRs2KCMjQxMmTNCDDz6oRo0aacuWLRo5cqR+//vf61//+tdl5wCgmhkAUAu9++67Rv369Q273W4UFxcb9erVM44cOWIsWrTI6N69u2EYhpGTk2NIMv7xj38Ya9euNSQZx44dMwzDMObNm2dIMr799ltzzMzMTCMkJMR83axZM2P69Onma7vdbjRv3tzo16+fue2vf/2r0ahRI8PPz8+46667jLS0NOOrr75ymKskY+TIkQ7bYmJijFGjRhmGYRj/93//Z7Rp08YoLy8395eUlBj+/v7GqlWrDMMwjNtuu81YtGiRwxhTpkwxLBaLYRiG8fbbbxtNmjQxzp49a+5/6623DEnGzp07DcMwjFWrVhmenp7GokWLjPPnzxv/+te/jHvvvdeQVGnsC8XHxxt/+MMfzNf33Xefcc8995ivz58/b9SvX98YOnSoue3w4cOGJCM3N/eS4wIAAFwMdd7V13kVBg8ebPj7+xuSjIceesjhPRMmTDC8vLyMf/3rX+a2zz77zPD09DQOHz5sGIZhJCQkGC1btjTKysrMmDZt2hj33nuv+bqi9vvLX/5iAKg53IEHoFbq0aOHTp8+rW3btumLL77QL37xC91888267777zOejrFu3TrfeeqtatGhx0TECAgJ02223ma+bNWumI0eOSJJOnDihw4cPKyYmxtxfr149RUdHO4wxcOBAHTp0SJ988onuv/9+rVu3Tl27dtX8+fMd4iwWS6XXFf8y+9VXX+nbb79Vw4YNzX/tbdy4sc6dO6fvvvtOp0+f1nfffafhw4eb+xs0aKAXX3xR3333nSRp37596tixo8Mddj8/ZlxcnF599VWNHDlSvr6++sUvfqEHHnhA0k/PTZF++jrJlClT1KFDBzVu3FgNGjTQqlWrVFBQ4DBWx44dzb97eXmpSZMm6tChg7ktJCREksx8AgAAVBV13tXXeRVmzpypHTt26G9/+5u+++47paamOuxv0aKFbrnlFodxysvLdeDAAXPbHXfcYdaG0k913YV1XkXtR50H1CwWsQBQK7Vu3VrNmzfX2rVrdezYMfPrAWFhYQoPD9emTZu0du1a9erV65JjeHt7O7z28PBwePZJVfn5+alPnz7q06ePxo8fryeffFITJkzQ7373uyq9/9SpU4qKitLChQsr7bv55pt16tQpSdK7777rUGhKPxVQVyM1NVUpKSk6fPiwGjVqpB9++EFpaWm69dZbJUmvvvqqZs+erVmzZqlDhw6qX7++kpOTza94VLhY7i7c5uHhIUkqLy+/qvkBAABQ5/3kaus8SQoNDVVoaKjatm2rxo0b695779X48ePVrFmzKo9xpTqvYht1HlCzuAMPQK1V8dDidevWqUePHub27t2767PPPtPWrVsv+VyUKwkKClKzZs20ZcsWc9v58+fNZ8VdTmRkpE6fPu2w7cIHLle8bteunSSpa9eu+uabb9S0aVO1bt3a4U9QUJBCQkIUFham77//vtL+iIgISVK7du3097//XefOnbvkMSt4eHgoLCxM/v7++stf/qLw8HB17dpVkvTll1+qX79+euyxx9SpUyfdeuut+vrrr6uQMQAAgOpDnXdtdd6FKhpsFz7Dr6CgQIcOHXIYx9PT02kr+QKoPjTwANRaPXv21MaNG5Wfn+/wgN777rtPb7/9tkpLS6+5sJOkZ599VtOmTdOyZcu0f/9+Pf300+bqZtJPK7b26tVLH3zwgf7+97/r4MGDWrp0qaZPn65+/fo5jLV06VLNnTtXX3/9tSZMmKCtW7cqKSlJkjRkyBDddNNN6tevn7744gsdPHhQ69at0zPPPGM+HHjSpEmaOnWqXn/9dX399dfatWuX5s2bp4yMDEnSb3/7W3l4eGjEiBHau3evVq5cqddee63SOb366qvatWuX9uzZoylTpmjatGl6/fXXzX/hvf3222Wz2bRp0ybt27dPv//971VUVHTNOQQAALgW1HlXV+etXLlS8+bN0+7du/XDDz9oxYoVGjlypO6++261atXKjPPz81NCQoK++uorffHFF3rmmWf0m9/8RqGhodecSwA1g6/QAqi1evbsqbNnz6pt27bmM9eknwq7kydPqk2bNlf1dYGf+8Mf/qDDhw8rISFBnp6eeuKJJ/SrX/1KJ06ckPTT6mQxMTGaOXOmvvvuO9ntdoWHh2vEiBF64YUXHMaaNGmSPvzwQz399NNq1qyZ/vKXvygyMlLST89o2bBhg5577jkNGDBAJ0+e1C233KLevXsrMDBQkvTkk08qICBAr776qsaMGaP69eurQ4cOSk5ONufy6aefauTIkerSpYsiIyP1yiuvaODAgQ7z+Oyzz/TSSy+ppKREnTp10t/+9jf17dvX3D9u3Dh9//33slqtCggI0FNPPaX+/fub5wwAAFATqPOurs7z9/fXu+++q5SUFJWUlCg8PFwDBgzQ888/7zDX1q1ba8CAAXrggQd09OhRPfjgg3rzzTevOY8Aao6HcS0PAgAAVJmHh4c+/vhj9e/f39VTAQAAQDWqTXXexIkTtWzZMuXn57t6KgCuAV+hBQAAAAAAANwYDTwAAAAAAADAjfEVWgAAAAAAAMCNcQceAAAAAAAA4MZo4AEAAAAAAABujAYeAAAAAAAA4MZo4AEAAAAAAABujAYeAAAAAAAA4MZo4AEAAAAAAABujAYeAAAAAAAA4MZo4AEAAAAAAABujAYeAAAAAAAA4Mb+Pw3LE+mInUc9AAAAAElFTkSuQmCC"/>

4개의 변수가 모두 왜곡되어 있음을 알 수 있다. 따라서 이상값을 찾기 위해 양자간 범위를 사용한다.



```python
# find outliers for Rainfall variable

IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

<pre>
Rainfall outliers are values < -2.4000000000000004 or > 3.2
</pre>
`Rainfall`의 경우 최소값과 최대값은 0.0과 371.0이다. 따라서 이상값은 값 > 3.2입니다.



```python
# find outliers for Evaporation variable

IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)
print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

<pre>
Evaporation outliers are values < -11.800000000000002 or > 21.800000000000004
</pre>
`Evaporation`의 경우 최소값과 최대값은 0.0과 145.0이다. 따라서 이상치는 값 > 21.8이다.



```python
# find outliers for WindSpeed9am variable

IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)
print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

<pre>
WindSpeed9am outliers are values < -29.0 or > 55.0
</pre>
`WindSpeed9am`의 경우 최소값과 최대값은 0.0과 130.0입니다. 따라서 이상값은 값 > 55.0입니다.



```python
# find outliers for WindSpeed3pm variable

IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)
print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

<pre>
WindSpeed3pm outliers are values < -20.0 or > 57.0
</pre>
`WindSpeed3pm`의 경우 최소값과 최대값은 0.0과 87.0이다. 따라서 이상값은 값 > 57.0이다.


# **8. 특징 벡터 및 대상 변수 선언** <a class="anchor" id="8"></a>



```python
X = df.drop(['RainTomorrow'], axis=1)

y = df['RainTomorrow']
```

# **9. 데이터를 별도의 훈련 및 테스트 세트로 분할** <a class="anchor" id="9"></a>




```python
# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```


```python
# check the shape of X_train and X_test

X_train.shape, X_test.shape
```

<pre>
((116368, 24), (29092, 24))
</pre>
# **10. 기능 엔지니어링** <a class="anchor" id="10"></a>



**기능 엔지니어링**은 원시 데이터를 모델을 더 잘 이해하고 예측력을 높이는 데 도움이 되는 유용한 기능으로 변환하는 프로세스이다. 다양한 유형의 변수에 대해 기능 엔지니어링을 수행한다.



먼저 범주형 변수와 수치형 변수를 다시 따로 표시하겠다.



```python
# check data types in X_train

X_train.dtypes
```

<pre>
Location          object
MinTemp          float64
MaxTemp          float64
Rainfall         float64
Evaporation      float64
Sunshine         float64
WindGustDir       object
WindGustSpeed    float64
WindDir9am        object
WindDir3pm        object
WindSpeed9am     float64
WindSpeed3pm     float64
Humidity9am      float64
Humidity3pm      float64
Pressure9am      float64
Pressure3pm      float64
Cloud9am         float64
Cloud3pm         float64
Temp9am          float64
Temp3pm          float64
RainToday         object
Year               int64
Month              int64
Day                int64
dtype: object
</pre>

```python
# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical
```

<pre>
['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
</pre>

```python
# display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical
```

<pre>
['MinTemp',
 'MaxTemp',
 'Rainfall',
 'Evaporation',
 'Sunshine',
 'WindGustSpeed',
 'WindSpeed9am',
 'WindSpeed3pm',
 'Humidity9am',
 'Humidity3pm',
 'Pressure9am',
 'Pressure3pm',
 'Cloud9am',
 'Cloud3pm',
 'Temp9am',
 'Temp3pm',
 'Year',
 'Month',
 'Day']
</pre>
**수치형 변수의 엔지니어링 결측값**






```python
# check missing values in numerical variables in X_train

X_train[numerical].isnull().sum()
```

<pre>
MinTemp           1183
MaxTemp           1019
Rainfall          2617
Evaporation      50355
Sunshine         55899
WindGustSpeed     8218
WindSpeed9am      1409
WindSpeed3pm      2456
Humidity9am       2147
Humidity3pm       3598
Pressure9am      12091
Pressure3pm      12064
Cloud9am         44796
Cloud3pm         47557
Temp9am           1415
Temp3pm           2865
Year                 0
Month                0
Day                  0
dtype: int64
</pre>

```python
# check missing values in numerical variables in X_test

X_test[numerical].isnull().sum()
```

<pre>
MinTemp            302
MaxTemp            242
Rainfall           644
Evaporation      12435
Sunshine         13936
WindGustSpeed     2045
WindSpeed9am       358
WindSpeed3pm       606
Humidity9am        507
Humidity3pm        909
Pressure9am       2974
Pressure3pm       2964
Cloud9am         11092
Cloud3pm         11801
Temp9am            352
Temp3pm            744
Year                 0
Month                0
Day                  0
dtype: int64
</pre>

```python
# print percentage of missing values in the numerical variables in training set

for col in numerical:
    if X_train[col].isnull().mean()>0:
        print(col, round(X_train[col].isnull().mean(),4))
```

<pre>
MinTemp 0.0102
MaxTemp 0.0088
Rainfall 0.0225
Evaporation 0.4327
Sunshine 0.4804
WindGustSpeed 0.0706
WindSpeed9am 0.0121
WindSpeed3pm 0.0211
Humidity9am 0.0185
Humidity3pm 0.0309
Pressure9am 0.1039
Pressure3pm 0.1037
Cloud9am 0.385
Cloud3pm 0.4087
Temp9am 0.0122
Temp3pm 0.0246
</pre>
**추정**



데이터가 완전히 무작위로 누락되었다고 가정한다(MCAR). 누락된 값을 대치하는 데 사용할 수 있는 두 가지 방법이 있다. 하나는 평균 또는 중앙값 전가이고 다른 하나는 무작위 표본 전가이다. 데이터 세트에 이상치가 있는 경우 중앙값 전가를 사용해야 한다. 따라서 중앙값 전가는 이상치에 강력하기 때문에 중앙값 전가를 사용하겠다.



데이터의 적절한 통계 측정치(이 경우 중앙값)로 누락된 값을 대치하겠다. 대체는 훈련 세트에 대해 수행된 다음 테스트 세트로 전파되어야 한다. 이는 열차 및 테스트 세트 모두에서 누락된 값을 채우는 데 사용되는 통계 측정이 열차 세트에서만 추출되어야 함을 의미한다. 이는 과적합을 방지하기 위한 것이다.



```python
# impute missing values in X_train and X_test with respective column median in X_train

for df1 in [X_train, X_test]:
    for col in numerical:
        col_median=X_train[col].median()
        df1[col].fillna(col_median, inplace=True)           
      
```


```python
# check again missing values in numerical variables in X_train

X_train[numerical].isnull().sum()
```

<pre>
MinTemp          0
MaxTemp          0
Rainfall         0
Evaporation      0
Sunshine         0
WindGustSpeed    0
WindSpeed9am     0
WindSpeed3pm     0
Humidity9am      0
Humidity3pm      0
Pressure9am      0
Pressure3pm      0
Cloud9am         0
Cloud3pm         0
Temp9am          0
Temp3pm          0
Year             0
Month            0
Day              0
dtype: int64
</pre>

```python
# check missing values in numerical variables in X_test

X_test[numerical].isnull().sum()
```

<pre>
MinTemp          0
MaxTemp          0
Rainfall         0
Evaporation      0
Sunshine         0
WindGustSpeed    0
WindSpeed9am     0
WindSpeed3pm     0
Humidity9am      0
Humidity3pm      0
Pressure9am      0
Pressure3pm      0
Cloud9am         0
Cloud3pm         0
Temp9am          0
Temp3pm          0
Year             0
Month            0
Day              0
dtype: int64
</pre>
이제 훈련 및 테스트 세트의 숫자 열에 누락된 값이 없음을 확인할 수 있다.


**범주형 변수의 엔지니어링 결측값**



```python
# print percentage of missing values in the categorical variables in training set

X_train[categorical].isnull().mean()
```

<pre>
Location       0.000000
WindGustDir    0.071068
WindDir9am     0.072597
WindDir3pm     0.028951
RainToday      0.022489
dtype: float64
</pre>

```python
# print categorical variables with missing data

for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))
```

<pre>
WindGustDir 0.07106764746322013
WindDir9am 0.07259727760208992
WindDir3pm 0.028951258077822083
RainToday 0.02248900041248453
</pre>

```python
# impute missing categorical variables with most frequent value

for df2 in [X_train, X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)
```


```python
# check missing values in categorical variables in X_train

X_train[categorical].isnull().sum()
```

<pre>
Location       0
WindGustDir    0
WindDir9am     0
WindDir3pm     0
RainToday      0
dtype: int64
</pre>

```python
# check missing values in categorical variables in X_test

X_test[categorical].isnull().sum()
```

<pre>
Location       0
WindGustDir    0
WindDir9am     0
WindDir3pm     0
RainToday      0
dtype: int64
</pre>
최종 확인으로 X_train 및 X_test에서 누락된 값을 확인한다.



```python
# check missing values in X_train

X_train.isnull().sum()
```

<pre>
Location         0
MinTemp          0
MaxTemp          0
Rainfall         0
Evaporation      0
Sunshine         0
WindGustDir      0
WindGustSpeed    0
WindDir9am       0
WindDir3pm       0
WindSpeed9am     0
WindSpeed3pm     0
Humidity9am      0
Humidity3pm      0
Pressure9am      0
Pressure3pm      0
Cloud9am         0
Cloud3pm         0
Temp9am          0
Temp3pm          0
RainToday        0
Year             0
Month            0
Day              0
dtype: int64
</pre>

```python
# check missing values in X_test

X_test.isnull().sum()
```

<pre>
Location         0
MinTemp          0
MaxTemp          0
Rainfall         0
Evaporation      0
Sunshine         0
WindGustDir      0
WindGustSpeed    0
WindDir9am       0
WindDir3pm       0
WindSpeed9am     0
WindSpeed3pm     0
Humidity9am      0
Humidity3pm      0
Pressure9am      0
Pressure3pm      0
Cloud9am         0
Cloud3pm         0
Temp9am          0
Temp3pm          0
RainToday        0
Year             0
Month            0
Day              0
dtype: int64
</pre>
X_train 및 X_test에 누락된 값이 없음을 확인할 수 있다.


**수치 변수의 엔지니어링 이상치**





`Rainfall`, `Evaporation`, `WindSpeed9am` 및 `WindSpeed3pm` 열에 이상값이 포함되어 있음을 확인했다. 나는 탑코딩 방식을 사용하여 최대값을 제한하고 위의 변수에서 이상값을 제거할 것이다.



```python
def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)
```


```python
X_train.Rainfall.max(), X_test.Rainfall.max()
```

<pre>
(3.2, 3.2)
</pre>

```python
X_train.Evaporation.max(), X_test.Evaporation.max()
```

<pre>
(21.8, 21.8)
</pre>

```python
X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()
```

<pre>
(55.0, 55.0)
</pre>

```python
X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()
```

<pre>
(57.0, 57.0)
</pre>

```python
X_train[numerical].describe()
```


  <div id="df-8b241286-9ef2-4c35-b4c1-13d1026da3d5">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>Pressure9am</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.190189</td>
      <td>23.203107</td>
      <td>0.670800</td>
      <td>5.093362</td>
      <td>7.982476</td>
      <td>39.982091</td>
      <td>14.029381</td>
      <td>18.687466</td>
      <td>68.950691</td>
      <td>51.605828</td>
      <td>1017.639891</td>
      <td>1015.244946</td>
      <td>4.664092</td>
      <td>4.710728</td>
      <td>16.979454</td>
      <td>21.657195</td>
      <td>2012.767058</td>
      <td>6.395091</td>
      <td>15.731954</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.366893</td>
      <td>7.085408</td>
      <td>1.181512</td>
      <td>2.800200</td>
      <td>2.761639</td>
      <td>13.127953</td>
      <td>8.835596</td>
      <td>8.700618</td>
      <td>18.811437</td>
      <td>20.439999</td>
      <td>6.728234</td>
      <td>6.661517</td>
      <td>2.280687</td>
      <td>2.106040</td>
      <td>6.449641</td>
      <td>6.848293</td>
      <td>2.538401</td>
      <td>3.425451</td>
      <td>8.796931</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-8.500000</td>
      <td>-4.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>980.500000</td>
      <td>977.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-7.200000</td>
      <td>-5.400000</td>
      <td>2007.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.700000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.200000</td>
      <td>31.000000</td>
      <td>7.000000</td>
      <td>13.000000</td>
      <td>57.000000</td>
      <td>37.000000</td>
      <td>1013.500000</td>
      <td>1011.100000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>12.300000</td>
      <td>16.700000</td>
      <td>2011.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>12.000000</td>
      <td>22.600000</td>
      <td>0.000000</td>
      <td>4.700000</td>
      <td>8.400000</td>
      <td>39.000000</td>
      <td>13.000000</td>
      <td>19.000000</td>
      <td>70.000000</td>
      <td>52.000000</td>
      <td>1017.600000</td>
      <td>1015.200000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>16.700000</td>
      <td>21.100000</td>
      <td>2013.000000</td>
      <td>6.000000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16.800000</td>
      <td>28.200000</td>
      <td>0.600000</td>
      <td>5.200000</td>
      <td>8.600000</td>
      <td>46.000000</td>
      <td>19.000000</td>
      <td>24.000000</td>
      <td>83.000000</td>
      <td>65.000000</td>
      <td>1021.800000</td>
      <td>1019.400000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>21.500000</td>
      <td>26.200000</td>
      <td>2015.000000</td>
      <td>9.000000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>31.900000</td>
      <td>48.100000</td>
      <td>3.200000</td>
      <td>21.800000</td>
      <td>14.500000</td>
      <td>135.000000</td>
      <td>55.000000</td>
      <td>57.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>1041.000000</td>
      <td>1039.600000</td>
      <td>9.000000</td>
      <td>8.000000</td>
      <td>40.200000</td>
      <td>46.700000</td>
      <td>2017.000000</td>
      <td>12.000000</td>
      <td>31.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8b241286-9ef2-4c35-b4c1-13d1026da3d5')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-8b241286-9ef2-4c35-b4c1-13d1026da3d5 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8b241286-9ef2-4c35-b4c1-13d1026da3d5');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  


이제 `Rainfall`, `Evaporation`, `WindSpeed9am` 및 `WindSpeed3pm` 열의 이상치가 제한되었음을 볼 수 있다.


**범주형 변수 인코딩**



```python
categorical
```

<pre>
['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
</pre>

```python
X_train[categorical].head()
```


  <div id="df-f0782d0e-d766-45ab-bfd7-24deed8e9e9d">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>WindGustDir</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>RainToday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22926</th>
      <td>NorfolkIsland</td>
      <td>ESE</td>
      <td>ESE</td>
      <td>ESE</td>
      <td>No</td>
    </tr>
    <tr>
      <th>80735</th>
      <td>Watsonia</td>
      <td>NE</td>
      <td>NNW</td>
      <td>NNE</td>
      <td>No</td>
    </tr>
    <tr>
      <th>121764</th>
      <td>Perth</td>
      <td>SW</td>
      <td>N</td>
      <td>SW</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>139821</th>
      <td>Darwin</td>
      <td>ESE</td>
      <td>ESE</td>
      <td>E</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1867</th>
      <td>Albury</td>
      <td>E</td>
      <td>ESE</td>
      <td>E</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-f0782d0e-d766-45ab-bfd7-24deed8e9e9d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-f0782d0e-d766-45ab-bfd7-24deed8e9e9d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-f0782d0e-d766-45ab-bfd7-24deed8e9e9d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  



```python
!pip install category_encoders
```

<pre>
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Requirement already satisfied: category_encoders in /usr/local/lib/python3.9/dist-packages (2.6.0)
Requirement already satisfied: scikit-learn>=0.20.0 in /usr/local/lib/python3.9/dist-packages (from category_encoders) (1.2.2)
Requirement already satisfied: pandas>=1.0.5 in /usr/local/lib/python3.9/dist-packages (from category_encoders) (1.5.3)
Requirement already satisfied: numpy>=1.14.0 in /usr/local/lib/python3.9/dist-packages (from category_encoders) (1.22.4)
Requirement already satisfied: patsy>=0.5.1 in /usr/local/lib/python3.9/dist-packages (from category_encoders) (0.5.3)
Requirement already satisfied: statsmodels>=0.9.0 in /usr/local/lib/python3.9/dist-packages (from category_encoders) (0.13.5)
Requirement already satisfied: scipy>=1.0.0 in /usr/local/lib/python3.9/dist-packages (from category_encoders) (1.10.1)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.9/dist-packages (from pandas>=1.0.5->category_encoders) (2022.7.1)
Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.9/dist-packages (from pandas>=1.0.5->category_encoders) (2.8.2)
Requirement already satisfied: six in /usr/local/lib/python3.9/dist-packages (from patsy>=0.5.1->category_encoders) (1.16.0)
Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.9/dist-packages (from scikit-learn>=0.20.0->category_encoders) (3.1.0)
Requirement already satisfied: joblib>=1.1.1 in /usr/local/lib/python3.9/dist-packages (from scikit-learn>=0.20.0->category_encoders) (1.2.0)
Requirement already satisfied: packaging>=21.3 in /usr/local/lib/python3.9/dist-packages (from statsmodels>=0.9.0->category_encoders) (23.0)
</pre>

```python
# encode RainToday variable

import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)
```


```python
X_train.head()
```


  <div id="df-3639183c-1070-4828-b9a0-114ad0185ef5">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Location</th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustDir</th>
      <th>WindGustSpeed</th>
      <th>WindDir9am</th>
      <th>WindDir3pm</th>
      <th>...</th>
      <th>Pressure3pm</th>
      <th>Cloud9am</th>
      <th>Cloud3pm</th>
      <th>Temp9am</th>
      <th>Temp3pm</th>
      <th>RainToday_0</th>
      <th>RainToday_1</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22926</th>
      <td>NorfolkIsland</td>
      <td>18.8</td>
      <td>23.7</td>
      <td>0.2</td>
      <td>5.0</td>
      <td>7.3</td>
      <td>ESE</td>
      <td>52.0</td>
      <td>ESE</td>
      <td>ESE</td>
      <td>...</td>
      <td>1013.9</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>21.4</td>
      <td>22.2</td>
      <td>0</td>
      <td>1</td>
      <td>2014</td>
      <td>3</td>
      <td>12</td>
    </tr>
    <tr>
      <th>80735</th>
      <td>Watsonia</td>
      <td>9.3</td>
      <td>24.0</td>
      <td>0.2</td>
      <td>1.6</td>
      <td>10.9</td>
      <td>NE</td>
      <td>48.0</td>
      <td>NNW</td>
      <td>NNE</td>
      <td>...</td>
      <td>1014.6</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>14.3</td>
      <td>23.2</td>
      <td>0</td>
      <td>1</td>
      <td>2016</td>
      <td>10</td>
      <td>6</td>
    </tr>
    <tr>
      <th>121764</th>
      <td>Perth</td>
      <td>10.9</td>
      <td>22.2</td>
      <td>1.4</td>
      <td>1.2</td>
      <td>9.6</td>
      <td>SW</td>
      <td>26.0</td>
      <td>N</td>
      <td>SW</td>
      <td>...</td>
      <td>1014.9</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>16.6</td>
      <td>21.5</td>
      <td>1</td>
      <td>0</td>
      <td>2011</td>
      <td>8</td>
      <td>31</td>
    </tr>
    <tr>
      <th>139821</th>
      <td>Darwin</td>
      <td>19.3</td>
      <td>29.9</td>
      <td>0.0</td>
      <td>9.2</td>
      <td>11.0</td>
      <td>ESE</td>
      <td>43.0</td>
      <td>ESE</td>
      <td>E</td>
      <td>...</td>
      <td>1012.1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>23.2</td>
      <td>29.1</td>
      <td>0</td>
      <td>1</td>
      <td>2010</td>
      <td>6</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1867</th>
      <td>Albury</td>
      <td>15.7</td>
      <td>17.6</td>
      <td>3.2</td>
      <td>4.7</td>
      <td>8.4</td>
      <td>E</td>
      <td>20.0</td>
      <td>ESE</td>
      <td>E</td>
      <td>...</td>
      <td>1010.5</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>16.5</td>
      <td>17.3</td>
      <td>1</td>
      <td>0</td>
      <td>2014</td>
      <td>4</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-3639183c-1070-4828-b9a0-114ad0185ef5')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-3639183c-1070-4828-b9a0-114ad0185ef5 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-3639183c-1070-4828-b9a0-114ad0185ef5');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  


`RainToday` 변수에서 `RainToday_0`과 `RainToday_1` 두 개의 추가 변수가 생성된 것을 확인할 수 있다.



이제 `X_train` 훈련 세트를 만들자.



```python
X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location), 
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)
```


```python
X_train.head()
```


  <div id="df-bc62729b-8979-4ab1-b7e3-9bdb88f96b4b">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22926</th>
      <td>18.8</td>
      <td>23.7</td>
      <td>0.2</td>
      <td>5.0</td>
      <td>7.3</td>
      <td>52.0</td>
      <td>31.0</td>
      <td>28.0</td>
      <td>74.0</td>
      <td>73.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>80735</th>
      <td>9.3</td>
      <td>24.0</td>
      <td>0.2</td>
      <td>1.6</td>
      <td>10.9</td>
      <td>48.0</td>
      <td>13.0</td>
      <td>24.0</td>
      <td>74.0</td>
      <td>55.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>121764</th>
      <td>10.9</td>
      <td>22.2</td>
      <td>1.4</td>
      <td>1.2</td>
      <td>9.6</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>85.0</td>
      <td>47.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>139821</th>
      <td>19.3</td>
      <td>29.9</td>
      <td>0.0</td>
      <td>9.2</td>
      <td>11.0</td>
      <td>43.0</td>
      <td>26.0</td>
      <td>17.0</td>
      <td>44.0</td>
      <td>37.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1867</th>
      <td>15.7</td>
      <td>17.6</td>
      <td>3.2</td>
      <td>4.7</td>
      <td>8.4</td>
      <td>20.0</td>
      <td>11.0</td>
      <td>13.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 118 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-bc62729b-8979-4ab1-b7e3-9bdb88f96b4b')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-bc62729b-8979-4ab1-b7e3-9bdb88f96b4b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-bc62729b-8979-4ab1-b7e3-9bdb88f96b4b');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  


마찬가지로 `X_test` 테스트 세트를 생성한다.



```python
X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_test.Location), 
                     pd.get_dummies(X_test.WindGustDir),
                     pd.get_dummies(X_test.WindDir9am),
                     pd.get_dummies(X_test.WindDir3pm)], axis=1)
```


```python
X_test.head()
```


  <div id="df-746679ae-1101-46f1-93d3-d84d95d0f960">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>138175</th>
      <td>21.9</td>
      <td>39.4</td>
      <td>1.6</td>
      <td>11.2</td>
      <td>11.5</td>
      <td>57.0</td>
      <td>20.0</td>
      <td>33.0</td>
      <td>50.0</td>
      <td>26.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38638</th>
      <td>20.5</td>
      <td>37.5</td>
      <td>0.0</td>
      <td>9.2</td>
      <td>8.4</td>
      <td>59.0</td>
      <td>17.0</td>
      <td>20.0</td>
      <td>47.0</td>
      <td>22.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>124058</th>
      <td>5.1</td>
      <td>17.2</td>
      <td>0.2</td>
      <td>4.7</td>
      <td>8.4</td>
      <td>50.0</td>
      <td>28.0</td>
      <td>22.0</td>
      <td>68.0</td>
      <td>51.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99214</th>
      <td>11.9</td>
      <td>16.8</td>
      <td>1.0</td>
      <td>4.7</td>
      <td>8.4</td>
      <td>28.0</td>
      <td>11.0</td>
      <td>13.0</td>
      <td>80.0</td>
      <td>79.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25097</th>
      <td>7.5</td>
      <td>21.3</td>
      <td>0.0</td>
      <td>4.7</td>
      <td>8.4</td>
      <td>15.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>88.0</td>
      <td>52.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 118 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-746679ae-1101-46f1-93d3-d84d95d0f960')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-746679ae-1101-46f1-93d3-d84d95d0f960 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-746679ae-1101-46f1-93d3-d84d95d0f960');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  


이제 모델 구축을 위한 교육 및 테스트 세트가 준비되었다. 그 전에 모든 기능 변수를 동일한 척도에 매핑해야 한다. 이를 `feature scaling`이라고 한다. 다음과 같이 하겠다.


**타깃 데이터 결측값**



```python
for df in [y_train, y_test]:
    mode = df.mode()[0]
    df.fillna(mode, inplace=True)
```

타깃 데이터의 결측값 또한 최빈값으로 대체한다.



```python
y_train.isna().sum()
```

<pre>
0
</pre>

```python
y_test.isna().sum()
```

<pre>
0
</pre>
타깃 데이터의 결측값을 제대로 처리했음을 알 수 있다.


# **11. 변수 크기 조정** <a class="anchor" id="11"></a>






```python
X_train.describe()
```


  <div id="df-92024b60-0e12-4d25-88bb-3b6d99c6f80a">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>...</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.190189</td>
      <td>23.203107</td>
      <td>0.670800</td>
      <td>5.093362</td>
      <td>7.982476</td>
      <td>39.982091</td>
      <td>14.029381</td>
      <td>18.687466</td>
      <td>68.950691</td>
      <td>51.605828</td>
      <td>...</td>
      <td>0.054078</td>
      <td>0.059123</td>
      <td>0.068447</td>
      <td>0.103723</td>
      <td>0.065224</td>
      <td>0.056055</td>
      <td>0.064786</td>
      <td>0.069323</td>
      <td>0.060309</td>
      <td>0.064958</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.366893</td>
      <td>7.085408</td>
      <td>1.181512</td>
      <td>2.800200</td>
      <td>2.761639</td>
      <td>13.127953</td>
      <td>8.835596</td>
      <td>8.700618</td>
      <td>18.811437</td>
      <td>20.439999</td>
      <td>...</td>
      <td>0.226173</td>
      <td>0.235855</td>
      <td>0.252512</td>
      <td>0.304902</td>
      <td>0.246922</td>
      <td>0.230029</td>
      <td>0.246149</td>
      <td>0.254004</td>
      <td>0.238059</td>
      <td>0.246452</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-8.500000</td>
      <td>-4.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.700000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.200000</td>
      <td>31.000000</td>
      <td>7.000000</td>
      <td>13.000000</td>
      <td>57.000000</td>
      <td>37.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>12.000000</td>
      <td>22.600000</td>
      <td>0.000000</td>
      <td>4.700000</td>
      <td>8.400000</td>
      <td>39.000000</td>
      <td>13.000000</td>
      <td>19.000000</td>
      <td>70.000000</td>
      <td>52.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16.800000</td>
      <td>28.200000</td>
      <td>0.600000</td>
      <td>5.200000</td>
      <td>8.600000</td>
      <td>46.000000</td>
      <td>19.000000</td>
      <td>24.000000</td>
      <td>83.000000</td>
      <td>65.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>31.900000</td>
      <td>48.100000</td>
      <td>3.200000</td>
      <td>21.800000</td>
      <td>14.500000</td>
      <td>135.000000</td>
      <td>55.000000</td>
      <td>57.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 118 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-92024b60-0e12-4d25-88bb-3b6d99c6f80a')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-92024b60-0e12-4d25-88bb-3b6d99c6f80a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-92024b60-0e12-4d25-88bb-3b6d99c6f80a');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  



```python
cols = X_train.columns
```


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
```


```python
X_train = pd.DataFrame(X_train, columns=[cols])
```


```python
X_test = pd.DataFrame(X_test, columns=[cols])
```


```python
X_train.describe()
```


  <div id="df-4afc387e-a341-4c18-879a-b1dc916a0177">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>MinTemp</th>
      <th>MaxTemp</th>
      <th>Rainfall</th>
      <th>Evaporation</th>
      <th>Sunshine</th>
      <th>WindGustSpeed</th>
      <th>WindSpeed9am</th>
      <th>WindSpeed3pm</th>
      <th>Humidity9am</th>
      <th>Humidity3pm</th>
      <th>...</th>
      <th>NNW</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SSE</th>
      <th>SSW</th>
      <th>SW</th>
      <th>W</th>
      <th>WNW</th>
      <th>WSW</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>...</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
      <td>116368.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.512133</td>
      <td>0.529359</td>
      <td>0.209625</td>
      <td>0.233640</td>
      <td>0.550516</td>
      <td>0.263427</td>
      <td>0.255080</td>
      <td>0.327850</td>
      <td>0.689507</td>
      <td>0.516058</td>
      <td>...</td>
      <td>0.054078</td>
      <td>0.059123</td>
      <td>0.068447</td>
      <td>0.103723</td>
      <td>0.065224</td>
      <td>0.056055</td>
      <td>0.064786</td>
      <td>0.069323</td>
      <td>0.060309</td>
      <td>0.064958</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.157596</td>
      <td>0.133940</td>
      <td>0.369223</td>
      <td>0.128450</td>
      <td>0.190458</td>
      <td>0.101767</td>
      <td>0.160647</td>
      <td>0.152642</td>
      <td>0.188114</td>
      <td>0.204400</td>
      <td>...</td>
      <td>0.226173</td>
      <td>0.235855</td>
      <td>0.252512</td>
      <td>0.304902</td>
      <td>0.246922</td>
      <td>0.230029</td>
      <td>0.246149</td>
      <td>0.254004</td>
      <td>0.238059</td>
      <td>0.246452</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.400990</td>
      <td>0.431002</td>
      <td>0.000000</td>
      <td>0.183486</td>
      <td>0.565517</td>
      <td>0.193798</td>
      <td>0.127273</td>
      <td>0.228070</td>
      <td>0.570000</td>
      <td>0.370000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.507426</td>
      <td>0.517958</td>
      <td>0.000000</td>
      <td>0.215596</td>
      <td>0.579310</td>
      <td>0.255814</td>
      <td>0.236364</td>
      <td>0.333333</td>
      <td>0.700000</td>
      <td>0.520000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.626238</td>
      <td>0.623819</td>
      <td>0.187500</td>
      <td>0.238532</td>
      <td>0.593103</td>
      <td>0.310078</td>
      <td>0.345455</td>
      <td>0.421053</td>
      <td>0.830000</td>
      <td>0.650000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 118 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-4afc387e-a341-4c18-879a-b1dc916a0177')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-4afc387e-a341-4c18-879a-b1dc916a0177 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-4afc387e-a341-4c18-879a-b1dc916a0177');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  


이제 로지스틱 회귀 분류기에 공급할 `X_train` 데이터 세트가 준비되었다.


# **12. 모델 훈련** <a class="anchor" id="12"></a>






```python
# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression


# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)


# fit the model
logreg.fit(X_train, y_train)
```

<style>#sk-container-id-6 {color: black;background-color: white;}#sk-container-id-6 pre{padding: 0;}#sk-container-id-6 div.sk-toggleable {background-color: white;}#sk-container-id-6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-6 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-6 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-6 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-6 div.sk-item {position: relative;z-index: 1;}#sk-container-id-6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-6 div.sk-item::before, #sk-container-id-6 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-6 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-6 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-6 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-6 div.sk-label-container {text-align: center;}#sk-container-id-6 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-6 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(random_state=0, solver=&#x27;liblinear&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" checked><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(random_state=0, solver=&#x27;liblinear&#x27;)</pre></div></div></div></div></div>


# **13. 결과 예측** <a class="anchor" id="13"></a>




```python
y_pred_test = logreg.predict(X_test)

y_pred_test
```

<pre>
array(['No', 'No', 'No', ..., 'No', 'No', 'No'], dtype=object)
</pre>
**predict_proba 메서드**





**predict_proba** 메서드는 이 경우 대상 변수(0과 1)에 대한 확률을 배열 형식으로 제공한다.



'0은 비가 오지 않을 확률'이고 '1은 비가 올 확률이다..



```python
# probability of getting output as 0 - no rain

logreg.predict_proba(X_test)[:,0]
```

<pre>
array([0.82582307, 0.79502108, 0.86451578, ..., 0.53041267, 0.74113011,
       0.9710027 ])
</pre>

```python
# probability of getting output as 1 - rain

logreg.predict_proba(X_test)[:,1]
```

<pre>
array([0.17417693, 0.20497892, 0.13548422, ..., 0.46958733, 0.25886989,
       0.0289973 ])
</pre>
# **14. 정확도 점수 확인** <a class="anchor" id="14"></a>



```python
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
```

<pre>
Model accuracy score: 0.8456
</pre>
여기서 **y_test**는 실제 클래스 레이블이고 **y_pred_test**는 테스트 세트에서 예측된 클래스 레이블이다.


**훈련 세트와 테스트 세트 정확도 비교**





이제 훈련 세트와 테스트 세트 정확도를 비교하여 과적합 여부를 확인하겠다.



```python
y_pred_train = logreg.predict(X_train)

y_pred_train
```

<pre>
array(['No', 'No', 'No', ..., 'No', 'No', 'No'], dtype=object)
</pre>

```python
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
```

<pre>
Training-set accuracy score: 0.8449
</pre>
**과적합 및 과소적합 확인**



```python
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))
```

<pre>
Training set score: 0.8449
Test set score: 0.8456
</pre>
훈련 세트 정확도 점수는 0.8449이고 테스트 세트 정확도는 0.8456이다. 이 두 값은 꽤 비슷하다. 따라서 과적합 문제는 없다.


로지스틱 회귀에서는 기본값 C = 1을 사용한다. 훈련 및 테스트 세트 모두에서 약 85%의 정확도로 우수한 성능을 제공한다. 그러나 훈련 세트와 테스트 세트의 모델 성능은 매우 비슷하다. 과소적합의 경우일 가능성이 높다.



나는 C를 높이고 더 유연한 모델에 맞출 것이다.



```python
# fit the Logsitic Regression model with C=100

# instantiate the model
logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)


# fit the model
logreg100.fit(X_train, y_train)
```

<style>#sk-container-id-7 {color: black;background-color: white;}#sk-container-id-7 pre{padding: 0;}#sk-container-id-7 div.sk-toggleable {background-color: white;}#sk-container-id-7 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-7 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-7 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-7 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-7 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-7 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-7 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-7 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-7 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-7 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-7 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-7 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-7 div.sk-item {position: relative;z-index: 1;}#sk-container-id-7 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-7 div.sk-item::before, #sk-container-id-7 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-7 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-7 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-7 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-7 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-7 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-7 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-7 div.sk-label-container {text-align: center;}#sk-container-id-7 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-7 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-7" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(C=100, random_state=0, solver=&#x27;liblinear&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" checked><label for="sk-estimator-id-9" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(C=100, random_state=0, solver=&#x27;liblinear&#x27;)</pre></div></div></div></div></div>



```python
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg100.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg100.score(X_test, y_test)))
```

<pre>
Training set score: 0.8458
Test set score: 0.8464
</pre>
우리는 C=100이 더 높은 테스트 세트 정확도와 훈련 세트 정확도를 약간 증가시킨다는 것을 볼 수 있다. 따라서 더 복잡한 모델이 더 잘 수행되어야 한다는 결론을 내릴 수 있다.


이제 C=0.01로 설정하여 기본값 C=1보다 더 정규화된 모델을 사용하면 어떻게 되는지 조사하겠다.



```python
# fit the Logsitic Regression model with C=001

# instantiate the model
logreg001 = LogisticRegression(C=0.01, solver='liblinear', random_state=0)


# fit the model
logreg001.fit(X_train, y_train)
```

<style>#sk-container-id-9 {color: black;background-color: white;}#sk-container-id-9 pre{padding: 0;}#sk-container-id-9 div.sk-toggleable {background-color: white;}#sk-container-id-9 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-9 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-9 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-9 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-9 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-9 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-9 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-9 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-9 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-9 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-9 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-9 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-9 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-9 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-9 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-9 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-9 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-9 div.sk-item {position: relative;z-index: 1;}#sk-container-id-9 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-9 div.sk-item::before, #sk-container-id-9 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-9 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-9 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-9 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-9 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-9 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-9 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-9 div.sk-label-container {text-align: center;}#sk-container-id-9 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-9 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-9" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(C=0.01, random_state=0, solver=&#x27;liblinear&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" checked><label for="sk-estimator-id-11" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(C=0.01, random_state=0, solver=&#x27;liblinear&#x27;)</pre></div></div></div></div></div>



```python
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg001.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg001.score(X_test, y_test)))
```

<pre>
Training set score: 0.8463
Test set score: 0.8463
</pre>
따라서 C=0.01로 설정하여 보다 정규화된 모델을 사용하면 교육 및 테스트 세트 정확도가 모두 기본 매개변수에 비해 감소한다.


**null 정확도와 모델 정확도 비교**





따라서 모델 정확도는 0.8501이다. 그러나 위의 정확도를 바탕으로 우리 모델이 매우 좋다고 말할 수는 없다. **null 정확도**와 비교해야 한다. Null 정확도는 항상 가장 빈번한 클래스를 예측하여 달성할 수 있는 정확도이다.



따라서 먼저 테스트 세트에서 클래스 분포를 확인해야 한다.



```python
# check class distribution in test set

y_test.value_counts()
```

<pre>
No     22726
Yes     6366
Name: RainTomorrow, dtype: int64
</pre>
가장 빈번한 클래스의 발생이 22726임을 알 수 있다. 따라서 22726을 총 발생 횟수로 나누어 null 정확도를 계산할 수 있다.



```python
# check null accuracy score

null_accuracy = (22726/(22726+6366))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))
```

<pre>
Null accuracy score: 0.7812
</pre>
모델 정확도 점수는 0.8501이지만 null 정확도 점수는 0.7812임을 알 수 있다. 따라서 로지스틱 회귀 모델이 클래스 레이블을 예측하는 데 매우 효과적이라는 결론을 내릴 수 있다.


이제 위의 분석을 기반으로 분류 모델 정확도가 매우 우수하다는 결론을 내릴 수 있다. 우리 모델은 클래스 레이블을 예측하는 측면에서 매우 훌륭하게 작동한다.





그러나 기본 값 분포는 제공하지 않는다. 또한 분류기가 만드는 오류 유형에 대해서는 아무 것도 알려주지 않는다.





우리는 우리를 구해줄 `Confusion matrix`라는 또 다른 도구가 있다.


# **15. 오차 행렬** <a class="anchor" id="15"></a>



오차 행렬은 분류 알고리즘의 성능을 요약하기 위한 도구이다. 오차차 행렬은 분류 모델 성능과 모델에서 생성된 오류 유형에 대한 명확한 그림을 제공한다. 각 범주별로 세분화된 올바른 예측과 잘못된 예측에 대한 요약을 제공한다. 요약은 표 형식으로 표시된다.





분류 모델 성능을 평가하는 동안 네 가지 유형의 결과가 가능하다. 이 네 가지 결과는 다음과 같다.





**참 긍정(TP)** – 참 긍정은 관찰이 특정 클래스에 속하고 관찰이 실제로 해당 클래스에 속한다고 예측할 때 발생한다.





**참음성(TN)** – 참음성은 관찰이 특정 클래스에 속하지 않는다고 예측하고 관찰이 실제로 해당 클래스에 속하지 않을 때 발생한다.





**거짓 양성(FP)** – 관측값이 특정 클래스에 속한다고 예측했지만 실제로는 해당 클래스에 속하지 않는 경우 거짓양성이 발생한다. 이러한 유형의 오류를 **1종 오류**라고 한다.







**거짓 음성(FN)** – 거짓 음성은 관측치가 특정 클래스에 속하지 않는다고 예측하지만 관측치가 실제로는 해당 클래스에 속할 때 발생한다. 이것은 매우 심각한 오류이며 **제2종 오류**라고 한한다.







이 네 가지 결과는 아래 주어진 혼동 매트릭스에 요약되어 있다.




```python
# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])
```

<pre>
Confusion matrix

 [[21555  1171]
 [ 3320  3046]]

True Positives(TP) =  21555

True Negatives(TN) =  3046

False Positives(FP) =  1171

False Negatives(FN) =  3320
</pre>


오차 행렬은 21555 + 3320 = 24875개의 올바른 예측과 3046 + 1171 = 4217개의 잘못된 예측을 보여준다.





이 경우, 우리는





- `True Positives` (실제 양성:1 및 예측 긍정:1) - 21555





- `True Negatives` (실제 음성:0 및 예측 네거티브:0) - 3046





- `False Positives` (실제제 음성:0이지만 양성 예측:1) - 1171 `(유형 I 오류)`





- `False Negatives` (실제 음성:1이지만 예측 부정:0) - 3320 `(유형 II 오류)`



```python
# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
```

<pre>
<Axes: >
</pre>
<img src="data:image/png;base64,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"/>

# **16. 분류 지표** <a class="anchor" id="16"></a>







**분류 보고서**





**분류 보고서**는 분류 모델 성능을 평가하는 또 다른 방법이다. 모델에 대한 **정밀도**, **리콜**, **f1** 및 **지원** 점수를 표시한다.



다음과 같이 분류 보고서를 인쇄할 수 있다.



```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))
```

<pre>
              precision    recall  f1-score   support

          No       0.87      0.95      0.91     22726
         Yes       0.72      0.48      0.58      6366

    accuracy                           0.85     29092
   macro avg       0.79      0.71      0.74     29092
weighted avg       0.83      0.85      0.83     29092

</pre>
**분류 정확도**



```python
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
```


```python
# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
```

<pre>
Classification accuracy : 0.8456
</pre>
**분류 오류**



```python
# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))
```

<pre>
Classification error : 0.1544
</pre>
**정밀도**





**정밀도**는 모든 예측된 긍정적 결과 중에서 올바르게 예측된 긍정적 결과의 백분율로 정의할 수 있다. 이는 참 양성(TP)과 참 양성 및 거짓 양성의 합(TP + FP)의 비율로 주어질 수 있다.





따라서 **정밀도**는 올바르게 예측된 양성 결과의 비율을 식별한다. 네거티브 클래스보다 포지티브 클래스에 더 관심이 있다.







수학적으로 정밀도는 `TP 대 (TP + FP)`의 비율로 정의할 수 있다.






```python
# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))
```

<pre>
Precision : 0.9485
</pre>
**재현율**





재현율은 모든 실제 긍정적 결과 중에서 올바르게 예측된 긍정적 결과의 백분율로 정의할 수 있다.

진양성과 거짓음성의 합(TP + FN)에 대한 진양성(TP)의 비율로 주어질 수 있다. **리콜**은 **민감도**라고도 한다.





**Recall**은 올바르게 예측된 실제 긍정의 비율을 식별한다.



수학적으로 재현율은 TP 대 `(TP + FN)`의 비율로 주어질 수 있다.



```python
recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))
```

<pre>
Recall or Sensitivity : 0.8665
</pre>
**참양성률**





**참양성률**은 **재현율**과 동의어이다.



```python
true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
```

<pre>
True Positive Rate : 0.8665
</pre>
**거짓 양성 비율**



```python
false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
```

<pre>
False Positive Rate : 0.2777
</pre>
**특이성**



```python
specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))
```

<pre>
Specificity : 0.7223
</pre>
**f1 점수**





**f1-점수**는 정밀도와 재현율의 가중 조화 평균이다. 가능한 최고의 **f1-점수**는 1.0이고 최악의 경우

0.0이 될 것이다. **f1-점수**는 정밀도와 재현율의 조화 평균이다. 따라서 **f1-점수**는 계산에 정밀도와 재현율을 포함하므로 항상 정확도 측정값보다 낮다. 'f1-score'의 가중 평균을 사용하여

전역 정확도가 아닌 분류기 모델을 비교한다.


**지원**





**지원**은 데이터 세트에서 해당 클래스의 실제 발생 횟수이다.


# **17. 임계값 수준 조정** <a class="anchor" id="17"></a>



```python
# print the first 10 predicted probabilities of two classes- 0 and 1

y_pred_prob = logreg.predict_proba(X_test)[0:10]

y_pred_prob
```

<pre>
array([[0.82582307, 0.17417693],
       [0.79502108, 0.20497892],
       [0.86451578, 0.13548422],
       [0.71329736, 0.28670264],
       [0.94347358, 0.05652642],
       [0.96973402, 0.03026598],
       [0.61694738, 0.38305262],
       [0.51709116, 0.48290884],
       [0.73924013, 0.26075987],
       [0.85797137, 0.14202863]])
</pre>
**관찰**





- 각 행에서 숫자의 합은 1이다.





- 2개의 클래스(0과 1)에 해당하는 2개의 열이 있다.



     - 등급 0 - 내일 비가 내리지 않을 것으로 예상되는 확률.

    

     - 등급 1 - 내일 비가 올 확률이 예측된다.

        

    

- 예측 확률의 중요성



     - 비가 올 확률 또는 비가 오지 않을 확률에 따라 관측치의 순위를 매길 수 있다.





- predict_proba 프로세스



     - 확률 예측

    

     - 확률이 가장 높은 클래스 선택

    

    

- 분류 임계값 수준



     - 0.5의 분류 임계값 수준이 있다.

    

     - 클래스 1 - 확률 > 0.5인 경우 강우 확률이 예측된다.

    

     - 클래스 0 - 확률 < 0.5인 경우 비가 내리지 않을 확률이 예측된다.



```python
# store the probabilities in dataframe

y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - No rain tomorrow (0)', 'Prob of - Rain tomorrow (1)'])

y_pred_prob_df
```


  <div id="df-f9e7d339-07cb-4708-8e75-a8d0ada50ffe">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Prob of - No rain tomorrow (0)</th>
      <th>Prob of - Rain tomorrow (1)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.825823</td>
      <td>0.174177</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.795021</td>
      <td>0.204979</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.864516</td>
      <td>0.135484</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.713297</td>
      <td>0.286703</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.943474</td>
      <td>0.056526</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.969734</td>
      <td>0.030266</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.616947</td>
      <td>0.383053</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.517091</td>
      <td>0.482909</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.739240</td>
      <td>0.260760</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.857971</td>
      <td>0.142029</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-f9e7d339-07cb-4708-8e75-a8d0ada50ffe')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-f9e7d339-07cb-4708-8e75-a8d0ada50ffe button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-f9e7d339-07cb-4708-8e75-a8d0ada50ffe');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  



```python
# print the first 10 predicted probabilities for class 1 - Probability of rain

logreg.predict_proba(X_test)[0:10, 1]
```

<pre>
array([0.17417693, 0.20497892, 0.13548422, 0.28670264, 0.05652642,
       0.03026598, 0.38305262, 0.48290884, 0.26075987, 0.14202863])
</pre>

```python
# store the predicted probabilities for class 1 - Probability of rain

y_pred1 = logreg.predict_proba(X_test)[:, 1]
```


```python
# plot histogram of predicted probabilities


# adjust the font size 
plt.rcParams['font.size'] = 12


# plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)


# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of rain')


# set the x-axis limit
plt.xlim(0,1)


# set the title
plt.xlabel('Predicted probabilities of rain')
plt.ylabel('Frequency')
```

<pre>
Text(0, 0.5, 'Frequency')
</pre>
<img src="data:image/png;base64,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gw8wfvx4vPXWWxp1FhYWyMnJ0blcVlYWLCwspHbAkyNbutoVblPSPnUJCQlBamqq9EhISHjGqyMiIqIXjUlFD8AQR48exejRoxEYGIjNmzdr1Ts7OyM/Px93797VOA2Zk5OD+/fvw8XFBQDg4OAApVKJ5ORkrT7UZeq2Je1TF6VSqfPoHBEREZFapT1ydu7cObzyyito06YN9u7dCxMT7Zzp5eUFAIiKitIoj4qKQkFBgVRvZGSEZs2aabVTr+ell16CjY2NXn0SERERlUalDGdXr15FYGAg3NzccODAgWJPJXbt2hUODg7YtGmTRvmmTZtgaWmJwMD/n5tr0KBBuHDhgkbo+vvvv/HTTz9h8ODBpeqTiIiISF+yO625YcMGPHr0SLrr8bvvvkNiYiIAYOrUqTAyMoK/vz8ePnyId999V2tesXr16qF9+/YAnlwftnDhQkyePBmDBw+Gv78/Tp06hV27dmHx4sVwcHCQlps0aRI++eQTBAYGYubMmTA1NcXq1atRs2ZNzJgxQ2qnT59ERERE+pLVLwQAgJubG27cuKGzLi4uDgDg7u5e7PJjxozR+j3OTz75BKtWrUJcXBxcXV0xZcoUTJ8+HQqFQqNdYmIi3n77bRw5cgQFBQXw8/PDmjVrUL9+fa31lLTPp+EvBBAREVU+5f0LAbILZy8ShjMiIqLK54X6+SYiIiKiFx3DGREREZGMMJwRERERyQjDGREREZGMMJwRERERyQjDGREREZGMMJwRERERyQjDGREREZGMMJwRERERyQjDGREREZGMMJwRERERyQjDGREREZGMMJwRERERyQjDGREREZGMMJwRERERyQjDGREREZGMMJwRERERyQjDGREREZGMMJwRERERyQjDGREREZGMMJwRERERyQjDGREREZGMMJwRERERyQjDGREREZGMMJwRERERyQjDGREREZGMMJwRERERyQjDGREREZGMMJwRERERyQjDGREREZGMMJwRERERyQjDGREREZGMMJwRERERyQjDGREREZGMMJwRERERyQjDGREREZGMMJwRERERyYjswll6ejpCQ0MREBAABwcHKBQKbNu2TWfbq1evIiAgANbW1nBwcMCoUaNw7949rXYFBQVYsWIF3N3dYW5ujubNm+PLL798bn0SERERlZRJRQ+gqJSUFCxYsAB16tRBixYtcOLECZ3tEhMT0blzZ6hUKixZsgTp6elYuXIl/vzzT5w/fx5mZmZS2/fffx/Lli3DG2+8gbZt2yIyMhLDhw+HQqHAsGHDyrVPIiIiIn0ohBCiogdRWHZ2Nh4+fAgnJydERUWhbdu2CA8Px9ixYzXaTZo0Cdu2bcO1a9dQp04dAMCxY8fQo0cPbNmyBUFBQQCAW7duwd3dHUFBQdiwYQMAQAgBX19fxMXFIT4+HsbGxuXW59OkpaVBpVLBNXgvjJSWhr95FSh+WWBFD4GIiOi5UH9/p6amwtbWtsz7l91pTaVSCScnp2e2++abb9CnTx8pRAFA9+7d0bBhQ+zdu1cqi4yMRG5uLiZNmiSVKRQKvPXWW0hMTMTZs2fLtU8iIiIifcgunJXErVu3cPfuXbRp00arztvbG9HR0dLz6OhoWFlZoUmTJlrt1PXl1ScRERGRvmR3zVlJJCcnAwCcnZ216pydnfHgwQNkZ2dDqVQiOTkZNWvWhEKh0GoHAElJSeXWZ1HZ2dnIzs6WnqelpZXo9RIREdGLo1IeOcvMzATw5BRoUebm5hptMjMzS9yurPssaunSpVCpVNLD1dW1uJdIREREL6hKGc4sLCwAQOMolFpWVpZGGwsLixK3K+s+iwoJCUFqaqr0SEhIKO4lEhER0QuqUoYz9elD9anIwpKTk+Hg4CAd2XJ2dsbt27dR9KZU9bIuLi7l1mdRSqUStra2Gg8iIiKiwiplOKtVqxYcHR0RFRWlVXf+/Hl4eXlJz728vJCRkYGrV69qtDt37pxUX159EhEREemrUoYzAHj11Vdx4MABjVODP/74I2JiYjB48GCprH///jA1NcXGjRulMiEENm/ejFq1aqFDhw7l2icRERGRPmR5t+aGDRvw6NEj6a7H7777DomJiQCAqVOnQqVSYc6cOfj666/RpUsXTJ8+Henp6QgLC0OzZs0wbtw4qa/atWsjODgYYWFhyM3NRdu2bbF//36cOnUKu3fv1pgstjz6JCIiItKH7H4hAADc3Nxw48YNnXVxcXFwc3MDAFy+fBnvvPMOTp8+DTMzMwQGBmLVqlWoWbOmxjIFBQVYvnw5tmzZguTkZDRo0AAhISEYMWKEVv/l0Wdx+AsBRERElU95/0KALMPZi4LhjIiIqPJ54X6+iYiIiOhFxnBGREREJCMMZ0REREQywnBGREREJCMMZ0REREQywnBGREREJCMMZ0REREQywnBGREREJCMMZ0REREQywnBGREREJCMMZ0REREQywnBGREREJCMMZ0REREQywnBGREREJCMMZ0REREQywnBGREREJCMMZ0REREQywnBGREREJCMMZ0REREQywnBGREREJCMMZ0REREQywnBGREREJCMMZ0REREQywnBGREREJCMMZ0REREQyYlA4S05OLqtxEBEREREMDGeurq7o2bMndu7cif/++6+sxkRERET0wjIonC1YsABJSUkYM2YMatasiZEjR+KHH35AQUFBWY2PiIiI6IViUDibM2cO/vrrL/z222948803ceLECfTu3RsuLi54++23ERUVVVbjJCIiInohlMkNAS1btsTKlSuRkJCAo0ePIjAwEOHh4fDx8UHTpk2xZMkS3Lx5syxWRURERFSllendmgqFAi+//DJ69+6Ndu3aQQiB2NhYzJs3Dy+99BIGDx7MmwiIiIiInqLMwtnx48cxYcIE1KxZE0OGDMHt27excuVKJCYmIjk5GcuWLcOPP/6IUaNGldUqiYiIiKocE0MW/v3337F79258+eWXSEpKgpOTEyZMmIDRo0ejWbNmGm1nzpwJc3NzzJw506ABExEREVVlBoWzli1bwsLCAgMGDMDo0aPRo0cPGBkVfzDOw8MD7du3N2SVRERERFWaQeHs888/x6BBg2BtbV2i9l26dEGXLl0MWSURERFRlWZQOBs7dmwZDYOIiIiIAANvCPjoo4/g7+9fbH2vXr2wadMmQ1ZBRERE9EIxKJx99tlnaNq0abH1TZs2xdatWw1ZBREREdELxaBwdv36dTRp0qTY+saNG+P69euGrKJYsbGxGDZsGGrXrg1LS0s0btwYCxYsQEZGhka7M2fOoFOnTrC0tISTkxOmTZuG9PR0rf6ys7Mxe/ZsuLi4wMLCAj4+Pjh69KjOdZe0TyIiIiJ9GXTNmZmZGW7fvl1sfXJy8lPv3iythIQEeHt7Q6VSYcqUKXBwcMDZs2cRGhqK3377DZGRkQCAS5cuoVu3bmjSpAlWr16NxMRErFy5ErGxsTh06JBGn2PHjkVERASCg4PRoEEDbNu2Db1798bx48fRqVMnqZ0+fRIRERHpy6Bw1q5dO2zbtg1vv/02bGxsNOpSU1MRHh6Odu3aGTRAXXbu3IlHjx7h9OnT8PDwAAAEBQWhoKAAO3bswMOHD2Fvb485c+bA3t4eJ06cgK2tLQDAzc0Nb7zxBo4cOYKePXsCAM6fP489e/YgLCxMmodt9OjR8PT0xKxZs3DmzBlp3SXtk4iIiKg0DDqsFRoaiqSkJHh5eWH9+vX46aef8NNPP+Gjjz5Cy5YtkZycjNDQ0LIaqyQtLQ0AULNmTY1yZ2dnGBkZwczMDGlpaTh69ChGjhwphSjgSeiytrbG3r17pbKIiAgYGxsjKChIKjM3N8f48eNx9uxZJCQkSOstaZ9EREREpWFQOPPx8cF3330HIQSmT5+OHj16oEePHggODoZCocC3335bLpPO+vn5AQDGjx+PS5cuISEhAV999RU2bdqEadOmwcrKCn/++Sfy8vLQpk0bjWXNzMzg5eWF6OhoqSw6OhoNGzbUCFwA4O3tDeDJqUwAevVJREREVBoGndYEgB49euCff/5BdHS0dPF/vXr10KpVKygUCoMHqEtAQAAWLlyIJUuW4Ntvv5XK33//fSxatAgApB9Yd3Z21lre2dkZp06dkp4nJycX2w4AkpKS9O5Tl+zsbGRnZ0vP1UcAiYiIiNQMDmcAYGRkhNatW6N169Zl0V2JuLm5oXPnznj11VdRrVo1fP/991iyZAmcnJwwZcoUZGZmAgCUSqXWsubm5lI9AGRmZhbbTl1f+L8l6VOXpUuXYv78+SV8hURERPQiKpNwduXKFfz77794+PAhhBBa9aNHjy6L1Uj27NmDoKAgxMTEoHbt2gCAgQMHoqCgALNnz8Zrr70GCwsLANA4UqWWlZUl1QOAhYVFse3U9YX/W5I+dQkJCcE777wjPU9LS4Orq+tTlyEiIqIXi0Hh7Pr16xg5ciTOnz+vM5QBgEKhKPNwtnHjRrRs2VIKZmr9+vXDtm3bEB0dLZ16VJ+KLCw5ORkuLi7Sc2dnZ9y6dUtnOwBSW3361EWpVOo86kZERESkZtANARMnTsSff/6JtWvX4uLFi4iLi9N6/Pvvv2U1VsmdO3eQn5+vVZ6bmwsAyMvLg6enJ0xMTBAVFaXRJicnB5cuXYKXl5dU5uXlhZiYGK1rwM6dOyfVA9CrTyIiIqLSMCic/fLLL5g9ezamTp0KLy8v1K1bV+ejrDVs2BDR0dGIiYnRKP/yyy9hZGSE5s2bQ6VSoXv37ti1axceP34stdm5cyfS09MxePBgqWzQoEHIz8/X+Kmp7OxshIeHw8fHRzr1qE+fRERERKVh0GnN6tWrQ6VSldVYSuzdd9/FoUOH8PLLL2PKlCmoVq0aDhw4gEOHDmHChAnS6cXFixejQ4cO8PX1RVBQEBITE7Fq1Sr07NkTAQEBUn8+Pj4YPHgwQkJCcPfuXdSvXx/bt29HfHw8PvvsM411l7RPIiIiotJQiOIuFiuBxYsXIzIyEmfPnoWxsXFZjuuZzp8/j3nz5iE6Ohr379+Hu7s7xowZg1mzZsHE5P8z5+nTpzF79mxcvHgRNjY2GDJkCJYuXar1iwZZWVmYO3cudu3ahYcPH6J58+ZYuHAh/P39tdZd0j6fJS0tDSqVCq7Be2GktCzdGyET8csCK3oIREREz4X6+zs1NVVrjtSyYFA4+/rrr7Fs2TJkZ2fj9ddfh6urq86QNnDgQIMGWVUxnBEREVU+5R3ODDqtOXToUOn/1b9JWZRCodB58T4RERERaTMonB0/frysxkFEREREMDCc+fr6ltU4iIiIiAhl9AsB2dnZuHjxIu7evYuOHTuievXqZdEtERER0QvHoHnOAOCjjz6Cs7MzOnXqhIEDB+KPP/4AAKSkpKB69er4/PPPDR4kERER0YvCoHAWHh6O4OBgBAQE4LPPPtP4Cafq1auja9eu2LNnj8GDJCIiInpRGBTOVq1ahf79++OLL75A3759tepbt26Ny5cvG7IKIiIioheKQeHsn3/+Qa9evYqtd3BwwP379w1ZBREREdELxaBwZmdnh5SUlGLrr1y5AicnJ0NWQURERPRCMSic9e7dG1u3bsWjR4+06i5fvoxPPvkE/fr1M2QVRERERC8Ug8LZokWLkJ+fD09PT3zwwQdQKBTYvn07Ro4ciTZt2qBGjRr48MMPy2qsRERERFWeQeHMxcUFv/32GwICAvDVV19BCIGdO3fiu+++w2uvvYZff/2Vc54RERER6cHgSWhr1KiBTz/9FJ9++inu3buHgoICODo6wsjI4CnUiIiIiF44ZfILAWqOjo5l2R0RERHRC8egcLZgwYJntlEoFJg7d64hqyEiIiJ6YShE4Wn99fS0U5cKhQJCCCgUCuTn55d2FVVaWloaVCoVXIP3wkhpWdHDMUj8ssCKHgIREdFzof7+Tk1Nha2tbZn3b9CFYQUFBVqPvLw8XL9+HW+//TbatGmDu3fvltVYiYiIiKq8Mr9q38jICO7u7li5ciUaNGiAqVOnlvUqiIiIiKqscr2lsnPnzjh48GB5roKIiIioSinXcBYVFcUpNYiIiIj0YNDdmjt27NBZ/ujRI/z888/43//+hwkTJhiyCiIiIqIXikHhbOzYscXWVa9eHe+99x5/vomIiIhIDwaFs7i4OK0yhUIBe3t72NjYGNI1ERER0QvJoHBWt27dshoHEREREaGcbwggIiIiIv0YdOTMyMgICoVCr2UUCgXy8vIMWS0RERFRlWVQOPvwww+xf/9+XL58Gf7+/mjUqBEA4Nq1azhy5Ag8PT0xYMCAshgnERER0QvBoHDm4uKCu3fv4q+//pKCmdrVq1fRtWtXuLi44I033jBokEREREQvCoOuOQsLC8OUKVO0ghkANGnSBFOmTMGKFSsMWQURERHRC8WgcJaYmAhTU9Ni601NTZGYmGjIKoiIiIheKAaFM09PT2zcuBG3bt3SqktMTMTGjRvRrFkzQ1ZBRERE9EIx6JqzNWvWwN/fHw0bNsQrr7yC+vXrAwBiY2Oxf/9+CCGwa9euMhkoERER0YvAoHDWqVMnnDt3DnPnzsW+ffuQmZkJALCwsIC/vz/mz5/PI2dEREREejAonAFPTm3u27cPBQUFuHfvHgDA0dERRkac35aIiIhIXwaHMzUjIyOYm5vD2tqawYyIiIiolAxOUVFRUQgICIClpSWqVauGkydPAgBSUlLQv39/nDhxwtBVEBEREb0wDApnZ86cQadOnRAbG4uRI0eioKBAqqtevTpSU1OxZcsWgwdJRERE9KIwKJzNmTMHTZo0wZUrV7BkyRKt+i5duuDcuXOGrOKpLl68iH79+sHBwQGWlpbw9PTERx99pNFGHSAtLS3h5OSEadOmIT09Xauv7OxszJ49Gy4uLrCwsICPjw+OHj2qc70l7ZOIiIhIXwaFswsXLmDcuHFQKpU6fwC9Vq1auH37tiGrKNaRI0fQvn173L17F3PnzsW6devQp08fjUlvL126hG7duiEjIwOrV6/GhAkTsHXrVgwePFirv7Fjx2L16tUYMWIE1q1bB2NjY/Tu3RunT5/WaKdPn0RERET6MuiGAFNTU41TmUXdunUL1tbWhqxCp7S0NIwePRqBgYGIiIgo9gaEOXPmwN7eHidOnICtrS0AwM3NDW+88QaOHDmCnj17AgDOnz+PPXv2ICwsDDNnzgQAjB49Gp6enpg1axbOnDmjd59EREREpWHQkbN27dohIiJCZ91///2H8PBw+Pr6GrIKnb744gvcuXMHixcvhpGREf777z+tkJiWloajR49i5MiRUogCnoQua2tr7N27VyqLiIiAsbExgoKCpDJzc3OMHz8eZ8+eRUJCgt59EhEREZWGQeFs/vz5iIqKQmBgIA4dOgQA+P333/Hpp5+idevWuHfvHubOnVsmAy3s2LFjsLW1xa1bt9CoUSNYW1vD1tYWb731FrKysgAAf/75J/Ly8tCmTRuNZc3MzODl5YXo6GipLDo6Gg0bNtQIXADg7e0N4MmpTH37JCIiIioNg8KZj48PDh48iH/++QejR48GAMyYMQNBQUHIz8/HwYMH0bx58zIZaGGxsbHIy8tD//794e/vj2+++Qavv/46Nm/ejHHjxgEAkpOTAQDOzs5ayzs7OyMpKUl6npycXGw7AFJbffrUJTs7G2lpaRoPIiIiosJKfc2ZEAKPHz9Ghw4d8Pfff+PSpUuIjY1FQUEB6tWrh9atW+u8SaAspKenIyMjA2+++aZ0d+bAgQORk5ODLVu2YMGCBdJPSSmVSq3lzc3NpXoAyMzMLLadur7wf0vSpy5Lly7F/PnzS/ISiYiI6AVV6iNnOTk5cHBwkMKRl5cXBg8ejKFDh6JNmzblFsyAJ7/dCQCvvfaaRvnw4cMBAGfPnpXaZGdnay2flZUl1av7K65d4fXp06cuISEhSE1NlR7qa9mIiIiI1EodzpRKJZycnHQeRSpvLi4uAICaNWtqlNeoUQMA8PDhQ+nUo/pUZGHJyclSH8CTU5LFtSu8Pn361EWpVMLW1lbjQURERFSYQdecjR07Fjt27EBOTk5ZjadEWrduDeDJVB2Fqa/5cnR0hKenJ0xMTBAVFaXRJicnB5cuXYKXl5dU5uXlhZiYGK1rwNQT6Krb6tMnERERUWkYFM6aNWuG7OxseHh4YPHixdi9ezf+97//aT3K2pAhQwAAn332mUb5p59+ChMTE/j5+UGlUqF79+7YtWsXHj9+LLXZuXMn0tPTNSaNHTRoEPLz87F161apLDs7G+Hh4fDx8YGrqysA6NUnERERUWkYNAlt4Wu+ipsyQ6FQID8/35DVaGnZsiVef/11fP7558jLy4Ovry9OnDiBr7/+GiEhIdLpxcWLF6NDhw7w9fVFUFAQEhMTsWrVKvTs2RMBAQFSfz4+Phg8eDBCQkJw9+5d1K9fH9u3b0d8fLxWACxpn0RERESloRBCCH0WmDNnDoYNG4bmzZvj5MmTJVqmPCaizc3NxZIlSxAeHo6kpCTUrVsXkydPRnBwsEa706dPY/bs2bh48SJsbGwwZMgQLF26FDY2NhrtsrKyMHfuXOzatQsPHz5E8+bNsXDhQvj7+2utu6R9PktaWhpUKhVcg/fCSGmp93sgJ/HLAit6CERERM+F+vs7NTW1XK4f1zucGRkZYdeuXdKdkffv30eNGjVw9OhRdO3atcwHWJUxnBEREVU+5R3ODLrmTE3PfEdERERExSiTcEZEREREZYPhjIiIiEhGSnW3Znx8PC5evAgASE1NBfDk9y7t7Ox0tm/VqlXpRkdERET0ginVDQFFf5pJCKHz55rU5WU9lUZVwRsCiIiIKp/yviFA7yNn4eHhZT4IIiIiInpC73A2ZsyY8hgHEREREYE3BBARERHJCsMZERERkYwwnBERERHJCMMZERERkYyUap4zoqLc3vu+oodQJjglCBERVTQeOSMiIiKSEYYzIiIiIhlhOCMiIiKSEYYzIiIiIhlhOCMiIiKSEYYzIiIiIhlhOCMiIiKSEYYzIiIiIhlhOCMiIiKSEYYzIiIiIhlhOCMiIiKSEYYzIiIiIhlhOCMiIiKSEYYzIiIiIhlhOCMiIiKSEYYzIiIiIhlhOCMiIiKSEYYzIiIiIhlhOCMiIiKSEYYzIiIiIhlhOCMiIiKSEYYzIiIiIhlhOCMiIiKSEYYzIiIiIhlhOCMiIiKSEYYzIiIiIhmpMuFs8eLFUCgU8PT01Ko7c+YMOnXqBEtLSzg5OWHatGlIT0/XapednY3Zs2fDxcUFFhYW8PHxwdGjR3Wur6R9EhEREemjSoSzxMRELFmyBFZWVlp1ly5dQrdu3ZCRkYHVq1djwoQJ2Lp1KwYPHqzVduzYsVi9ejVGjBiBdevWwdjYGL1798bp06dL3ScRERGRPkwqegBlYebMmWjXrh3y8/ORkpKiUTdnzhzY29vjxIkTsLW1BQC4ubnhjTfewJEjR9CzZ08AwPnz57Fnzx6EhYVh5syZAIDRo0fD09MTs2bNwpkzZ/Tuk4iIiEhflf7I2c8//4yIiAisXbtWqy4tLQ1Hjx7FyJEjpRAFPAld1tbW2Lt3r1QWEREBY2NjBAUFSWXm5uYYP348zp49i4SEBL37JCIiItJXpQ5n+fn5mDp1KiZMmIBmzZpp1f/555/Iy8tDmzZtNMrNzMzg5eWF6OhoqSw6OhoNGzbUCFwA4O3tDeDJqUx9+yQiIiLSV6U+rbl582bcuHEDx44d01mfnJwMAHB2dtaqc3Z2xqlTpzTaFtcOAJKSkvTus6js7GxkZ2dLz9PS0optS0RERC+mSnvk7P79+/jwww8xd+5cODo66myTmZkJAFAqlVp15ubmUr26bXHtCvelT59FLV26FCqVSnq4uroW25aIiIheTJU2nH3wwQdwcHDA1KlTi21jYWEBABpHq9SysrKkenXb4toV7kufPosKCQlBamqq9FBfx0ZERESkVilPa8bGxmLr1q1Yu3atdLoReBKOcnNzER8fD1tbW+nUo/pUZGHJyclwcXGRnjs7O+PWrVs62wGQ2urTZ1FKpVLnETciIiIitUp55OzWrVsoKCjAtGnT4O7uLj3OnTuHmJgYuLu7Y8GCBfD09ISJiQmioqI0ls/JycGlS5fg5eUllXl5eSEmJkbrOrBz585J9QD06pOIiIhIX5UynHl6emLfvn1aDw8PD9SpUwf79u3D+PHjoVKp0L17d+zatQuPHz+Wlt+5cyfS09M1Jo0dNGgQ8vPzsXXrVqksOzsb4eHh8PHxka4P06dPIiIiIn0phBCiogdRVvz8/JCSkoK//vpLKrt48SI6dOiApk2bIigoCImJiVi1ahU6d+6Mw4cPayw/ZMgQ7Nu3D2+//Tbq16+P7du34/z58/jxxx/RuXPnUvX5NGlpaU9uDAjeCyOlpeFvABksfllgRQ+BiIhkTv39nZqaqjUFV1molEfO9NGqVSscO3YMFhYWePvtt7F161aMHz8eERERWm137NiB4OBg7Ny5E9OmTUNubi4OHDigEcz07ZOIiIhIH1XqyFllwyNn8sMjZ0RE9Cw8ckZERET0AmE4IyIiIpIRhjMiIiIiGWE4IyIiIpIRhjMiIiIiGWE4IyIiIpIRhjMiIiIiGWE4IyIiIpIRhjMiIiIiGWE4IyIiIpIRhjMiIiIiGWE4IyIiIpIRhjMiIiIiGWE4IyIiIpIRhjMiIiIiGWE4IyIiIpIRhjMiIiIiGWE4IyIiIpIRhjMiIiIiGWE4IyIiIpIRhjMiIiIiGWE4IyIiIpIRk4oeAJGcuL33fUUPwWDxywIreghERGQAHjkjIiIikhGGMyIiIiIZYTgjIiIikhGGMyIiIiIZYTgjIiIikhGGMyIiIiIZYTgjIiIikhGGMyIiIiIZYTgjIiIikhGGMyIiIiIZYTgjIiIikhGGMyIiIiIZYTgjIiIikhGGMyIiIiIZqZTh7MKFC5gyZQo8PDxgZWWFOnXqYMiQIYiJidFqe/XqVQQEBMDa2hoODg4YNWoU7t27p9WuoKAAK1asgLu7O8zNzdG8eXN8+eWXOtdf0j6JiIiI9GVS0QMojeXLl+OXX37B4MGD0bx5c9y+fRsbNmxAq1at8Ouvv8LT0xMAkJiYiM6dO0OlUmHJkiVIT0/HypUr8eeff+L8+fMwMzOT+nz//fexbNkyvPHGG2jbti0iIyMxfPhwKBQKDBs2TGqnT59ERERE+lIIIURFD0JfZ86cQZs2bTSCUGxsLJo1a4ZBgwZh165dAIBJkyZh27ZtuHbtGurUqQMAOHbsGHr06IEtW7YgKCgIAHDr1i24u7sjKCgIGzZsAAAIIeDr64u4uDjEx8fD2NhYrz5LIi0tDSqVCq7Be2GktDT8jSECEL8ssKKHQERUpam/v1NTU2Fra1vm/VfK05odOnTQOkLVoEEDeHh44OrVq1LZN998gz59+kghCgC6d++Ohg0bYu/evVJZZGQkcnNzMWnSJKlMoVDgrbfeQmJiIs6ePat3n0RERESlUSnDmS5CCNy5cwfVq1cH8ORo2N27d9GmTRuttt7e3oiOjpaeR0dHw8rKCk2aNNFqp67Xt08iIiKi0qgy4Wz37t24desWhg4dCgBITk4GADg7O2u1dXZ2xoMHD5CdnS21rVmzJhQKhVY7AEhKStK7T12ys7ORlpam8SAiIiIqrEqEs2vXrmHy5Mlo3749xowZAwDIzMwEACiVSq325ubmGm0yMzNL3K6kfeqydOlSqFQq6eHq6lqyF0hEREQvjEofzm7fvo3AwECoVCpERERIF+5bWFgAgM4jWVlZWRptLCwsStyupH3qEhISgtTUVOmRkJBQshdJREREL4xKOZWGWmpqKnr16oVHjx7h1KlTcHFxkerUpx7VpyILS05OhoODg3QEzNnZGcePH4cQQuPUpnpZdb/69KmLUql8aj0RERFRpT1ylpWVhb59+yImJgYHDhxA06ZNNepr1aoFR0dHREVFaS17/vx5eHl5Sc+9vLyQkZGhcacnAJw7d06q17dPIiIiotKolOEsPz8fQ4cOxdmzZ/H111+jffv2Otu9+uqrOHDggMbpwx9//BExMTEYPHiwVNa/f3+Ymppi48aNUpkQAps3b0atWrXQoUMHvfskIiIiKo1KOQltcHAw1q1bh759+2LIkCFa9SNHjgQAJCQkoGXLlrCzs8P06dORnp6OsLAw1K5dGxcuXNA4xThr1iyEhYUhKCgIbdu2xf79+/H9999j9+7dGD58uNROnz6fhZPQEunGiXSJSM7KexLaShnO/Pz8cPLkyWLrC7+ky5cv45133sHp06dhZmaGwMBArFq1CjVr1tRYpqCgAMuXL8eWLVuQnJyMBg0aICQkBCNGjNDqv6R9PgvDGZFuDGdEJGcMZ1UYwxmRbgxnRCRn/PkmIiIiohcIwxkRERGRjDCcEREREckIwxkRERGRjDCcEREREckIwxkRERGRjDCcEREREckIwxkRERGRjDCcEREREckIwxkRERGRjDCcEREREcmISUUPgIioKLf3vq/oIZQJ/kYoEZUGj5wRERERyQjDGREREZGMMJwRERERyQjDGREREZGM8IYAIqJyUhVubOBNDUTPH4+cEREREckIwxkRERGRjDCcEREREckIrzkjIqJiVYXr5gBeO0eVC8MZERFVeVUhZDJgvjh4WpOIiIhIRhjOiIiIiGSEpzWJiIgqgapwahbg6dmS4JEzIiIiIhlhOCMiIiKSEYYzIiIiIhlhOCMiIiKSEd4QQERERM9NVbix4Y85L5dr/zxyRkRERCQjDGdEREREMsJwRkRERCQjDGdEREREMsJwRkRERCQjDGdEREREMsJwRkRERCQjDGdEREREMsJwVkrZ2dmYPXs2XFxcYGFhAR8fHxw9erSih0VERESVHMNZKY0dOxarV6/GiBEjsG7dOhgbG6N37944ffp0RQ+NiIiIKjH+fFMpnD9/Hnv27EFYWBhmzpwJABg9ejQ8PT0xa9YsnDlzpoJHSERERJUVj5yVQkREBIyNjREUFCSVmZubY/z48Th79iwSEhIqcHRERERUmTGclUJ0dDQaNmwIW1tbjXJvb28AwKVLlypgVERERFQV8LRmKSQnJ8PZ2VmrXF2WlJSkc7ns7GxkZ2dLz1NTUwEABdkZ5TBKIiIiKg9paWkAACFEufTPcFYKmZmZUCqVWuXm5uZSvS5Lly7F/PnztcpvbRpbpuMjIiKi8uO69sl/79+/D5VKVeb9M5yVgoWFhcYRMLWsrCypXpeQkBC888470vNHjx6hbt26uHnzZrlsXCq5tLQ0uLq6IiEhQet0NT1f3Bbywu0hH9wW8pGamoo6derAwcGhXPpnOCsFZ2dn3Lp1S6s8OTkZAODi4qJzOaVSqfOIm0ql4h+aTNja2nJbyAS3hbxwe8gHt4V8GBmVz6X7vCGgFLy8vBATEyOdc1Y7d+6cVE9ERERUGgxnpTBo0CDk5+dj69atUll2djbCw8Ph4+MDV1fXChwdERERVWY8rVkKPj4+GDx4MEJCQnD37l3Ur18f27dvR3x8PD777LMS96NUKhEaGqrzVCc9X9wW8sFtIS/cHvLBbSEf5b0tFKK87gOt4rKysjB37lzs2rULDx8+RPPmzbFw4UL4+/tX9NCIiIioEmM4IyIiIpIRXnNGREREJCMMZ0REREQywnBGREREJCMMZ+UgOzsbs2fPhouLCywsLODj44OjR4+WaNlbt25hyJAhsLOzg62tLfr3749///23nEdcdZV2W/zvf//D0KFD8dJLL8HS0hKNGjXCjBkz8OjRo/IfdBVlyN9FYT169IBCocCUKVPKYZQvDkO3x1dffYX27dvDysoKdnZ26NChA3766adyHHHVZci2OHbsGLp06YLq1avDzs4O3t7e2LlzZzmPuOpKT09HaGgoAgIC4ODgAIVCgW3btpV4+UePHiEoKAiOjo6wsrJCly5dcPHiRf0HIqjMDRs2TJiYmIiZM2eKLVu2iPbt2wsTExNx6tSppy73+PFj0aBBA1GjRg2xfPlysXr1auHq6ipq164tUlJSntPoq5bSbotq1aqJZs2aiblz54pPPvlETJs2TZiZmYnGjRuLjIyM5zT6qqW026Kwb775RlhZWQkAYvLkyeU42qrPkO0RGhoqFAqFGDx4sNi8ebNYv369mDhxotixY8dzGHnVU9ptERkZKRQKhejQoYNYv3692LBhg+jcubMAIFavXv2cRl+1xMXFCQCiTp06ws/PTwAQ4eHhJVo2Pz9fdOjQQVhZWYl58+aJDRs2iKZNmwobGxsRExOj1zgYzsrYuXPnBAARFhYmlWVmZop69eqJ9u3bP3XZ5cuXCwDi/PnzUtnVq1eFsbGxCAkJKbcxV1WGbIvjx49rlW3fvl0AEJ988klZD7XKM2RbFG7v5uYmFixYwHBmIEO2x9mzZ4VCoeCXfxkxZFv06NFDuLi4iKysLKksNzdX1KtXTzRv3rzcxlyVZWVlieTkZCGEEBcuXNArnH311VcCgPj666+lsrt37wo7Ozvx2muv6TUOntYsYxERETA2NkZQUJBUZm5ujvHjx+Ps2bNISEh46rJt27ZF27ZtpbLGjRujW7du2Lt3b7mOuyoyZFv4+flplb3yyisAgKtXr5b5WKs6Q7aF2ooVK1BQUICZM2eW51BfCIZsj7Vr18LJyQnTp0+HEALp6enPY8hVliHbIi0tDfb29hoToZqYmKB69eqwsLAo13FXVUqlEk5OTqVaNiIiAjVr1sTAgQOlMkdHRwwZMgSRkZHIzs4ucV8MZ2UsOjoaDRs21PpRWm9vbwDApUuXdC5XUFCAP/74A23atNGq8/b2xvXr1/H48eMyH29VVtptUZzbt28DAKpXr14m43uRGLotbt68iWXLlmH58uX80ikDhmyPH3/8EW3btsVHH30ER0dH2NjYwNnZGRs2bCjPIVdZhmwLPz8/XL58GXPnzsU///yD69evY+HChYiKisKsWbPKc9ikQ3R0NFq1aqX1Y+je3t7IyMhATExMifvizzeVseTkZDg7O2uVq8uSkpJ0LvfgwQNkZ2c/c9lGjRqV4WirttJui+IsX74cxsbGGDRoUJmM70Vi6LaYMWMGWrZsiWHDhpXL+F40pd0eDx8+REpKCn755Rf89NNPCA0NRZ06dRAeHo6pU6fC1NQUEydOLNexVzWG/G3MnTsXcXFxWLx4MRYtWgQAsLS0xDfffIP+/fuXz4CpWMnJyejcubNWeeFt2axZsxL1xXBWxjIzM3X+1pa5ublUX9xyAEq1LOlW2m2hyxdffIHPPvsMs2bNQoMGDcpsjC8KQ7bF8ePH8c033+DcuXPlNr4XTWm3h/oU5v3797Fnzx4MHToUADBo0CA0a9YMixYtYjjTkyF/G0qlEg0bNsSgQYMwcOBA5OfnY+vWrRg5ciSOHj2Kdu3aldu4SVtZfucwnJUxCwsLneeVs7KypPrilgNQqmVJt9Jui6JOnTqF8ePHw9/fH4sXLy7TMb4oSrst8vLyMG3aNIwaNUrjWkwyjKGfU6amphpHkI2MjDB06FCEhobi5s2bqFOnTjmMumoy5HNqypQp+PXXX3Hx4kXpVNqQIUPg4eGB6dOn8x80z1lZfecAvOaszDk7OyM5OVmrXF3m4uKiczkHBwcolcpSLUu6lXZbFPb777+jX79+8PT0REREBExM+O+Z0ijtttixYwf+/vtvTJw4EfHx8dIDAB4/foz4+HhkZGSU27irKkM+p8zNzVGtWjUYGxtr1NWoUQPAk1OfVHKl3RY5OTn47LPPEBgYqHGNk6mpKXr16oWoqCjk5OSUz6BJp7L4zlFjOCtjXl5eiImJQVpamka5+l8wXl5eOpczMjJCs2bNEBUVpVV37tw5vPTSS7CxsSnz8VZlpd0WatevX0dAQABq1KiBgwcPwtrauryGWuWVdlvcvHkTubm56NixI9zd3aUH8CS4ubu748iRI+U69qrIkM8pLy8v3Lt3T+uLX31tlKOjY9kPuAor7ba4f/8+8vLykJ+fr1WXm5uLgoICnXVUfry8vHDx4kUUFBRolJ87dw6WlpZo2LBhyTvTa+INeqZff/1Va86arKwsUb9+feHj4yOV3bhxQ1y9elVj2WXLlgkA4sKFC1LZtWvXhLGxsZg9e3b5D76KMWRbJCcni5deekm4uLiIuLi45zXkKqu02+Lq1ati3759Wg8Aonfv3mLfvn0iKSnpub6WqsCQv401a9YIAGLr1q1SWWZmpnjppZdE06ZNy3/wVUxpt0VeXp6ws7MTDRs2FNnZ2VL548ePRe3atUXjxo2fzwuowp42z1lSUpK4evWqyMnJkcr27NmjNc/ZvXv3hJ2dnRg6dKhe62Y4KweDBw8WJiYm4t133xVbtmwRHTp0ECYmJuLkyZNSG19fX1E0G6elpYl69eqJGjVqiBUrVog1a9YIV1dX4eLiIu7evfu8X0aVUNpt0aJFCwFAzJo1S+zcuVPjceTIkef9MqqE0m4LXcBJaA1W2u2RkZEhPDw8hKmpqZg5c6b46KOPRNu2bYWxsbE4ePDg834ZVUJpt8WiRYsEANGyZUuxZs0asXLlStGkSRMBQOzatet5v4wqY/369WLhwoXirbfeEgDEwIEDxcKFC8XChQvFo0ePhBBCjBkzRgDQ+Md7Xl6eaNeunbC2thbz588XH3/8sfDw8BA2Njbi2rVreo2B4awcZGZmipkzZwonJyehVCpF27ZtxQ8//KDRprgvoYSEBDFo0CBha2srrK2tRZ8+fURsbOzzGnqVU9ptAaDYh6+v73N8BVWHIX8XRTGcGc6Q7XHnzh0xZswY4eDgIJRKpfDx8dFalkrOkG2xe/du4e3tLezs7ISFhYXw8fERERERz2voVVLdunWL/fxXhzFd4UwIIR48eCDGjx8vqlWrJiwtLYWvr6/G2bCSUgghRMlPghIRERFReeINAUREREQywnBGREREJCMMZ0REREQywnBGREREJCMMZ0REREQywnBGREREJCMMZ0REREQywnBGREREJCMMZ0QvMDc3N4wdO1Z6fuLECSgUCpw4caLCxlRU0THKgUKhwJQpU8qsv23btkGhUCAqKuqZbf38/ODn5yc9j4+Ph0KhwLZt26SyefPmQaFQ6LXu+Ph4PUf9fFy4cAEdOnSAlZUVFAoFLl26VG7rkuP+Ty8mhjOiCqL+UlQ/zM3N0bBhQ0yZMgV37typ6OHp5eDBg5g3b15FD4OeYsmSJdi/f39FD0Mvubm5GDx4MB48eIA1a9Zg586dqFu3bkUPi6jcmVT0AIhedAsWLIC7uzuysrJw+vRpbNq0CQcPHsRff/0FS0vL5zqWzp07IzMzE2ZmZnotd/DgQXz88ccMaM/BkSNHntnmgw8+wHvvvadRtmTJEgwaNAgDBgzQKB81ahSGDRsGpVJZlsMsE9evX8eNGzfwySefYMKECeW+vtLu/0RljeGMqIL16tULbdq0AQBMmDAB1apVw+rVqxEZGYnXXntN5zL//fcfrKysynwsRkZGMDc3L/N+5a683s/yUJLgYGJiAhOTkn28Gxsbw9jY2NBhlYu7d+8CAOzs7Eq1vL7b9UXd/0l+eFqTSGa6du0KAIiLiwMAjB07FtbW1rh+/Tp69+4NGxsbjBgxAgBQUFCAtWvXwsPDA+bm5qhZsyYmTpyIhw8favQphMCiRYtQu3ZtWFpaokuXLrh8+bLWuou75ubcuXPo3bs37O3tYWVlhebNm2PdunXS+D7++GMA0DhNq1bWY9RFfd3VypUrsWbNGtStWxcWFhbw9fXFX3/9pdH2ae/nf//9hxkzZsDV1RVKpRKNGjXCypUrIYTQud7du3ejUaNGMDc3R+vWrfHzzz9r1N+4cQOTJk1Co0aNYGFhgWrVqmHw4MHFXt+VkZGBiRMnolq1arC1tcXo0aO13qei15zpUvSaM4VCgf/++w/bt2+Xto/6Or7irjk7dOgQXn75ZVhZWcHGxgaBgYFa2+P27dsYN24cateuDaVSCWdnZ/Tv379E16/99NNPUv92dnbo378/rl69KtWPHTsWvr6+AIDBgwdDoVA89XWrX8fJkycxadIk1KhRA7Vr1wZQ8u2ga//38/ODp6cnrly5gi5dusDS0hK1atXCihUrnvkaiUqLR86IZOb69esAgGrVqklleXl58Pf3R6dOnbBy5UrpdOfEiROxbds2jBs3DtOmTUNcXBw2bNiA6Oho/PLLLzA1NQUAfPjhh1i0aBF69+6N3r174+LFi+jZsydycnKeOZ6jR4+iT58+cHZ2xvTp0+Hk5ISrV6/iwIEDmD59OiZOnIikpCQcPXoUO3fu1Fr+eYxRbceOHXj8+DEmT56MrKwsrFu3Dl27dsWff/6JmjVrPvX9FEKgX79+OH78OMaPHw8vLy8cPnwY7777Lm7duoU1a9ZorOvkyZP46quvMG3aNCiVSmzcuBEBAQE4f/48PD09ATy5mP3MmTMYNmwYateujfj4eGzatAl+fn64cuWK1mnrKVOmwM7ODvPmzcPff/+NTZs24caNG1JoKK2dO3diwoQJ8Pb2RlBQEACgXr16T20/ZswY+Pv7Y/ny5cjIyMCmTZvQqVMnREdHw83NDQDw6quv4vLly5g6dSrc3Nxw9+5dHD16FDdv3pTa6HLs2DH06tULL730EubNm4fMzEysX78eHTt2xMWLF+Hm5oaJEyeiVq1aWLJkCaZNm4a2bdtqbMPiTJo0CY6Ojvjwww/x33//AdB/OxT18OFDBAQEYODAgRgyZAgiIiIwe/ZsNGvWDL169XrmmIj0JoioQoSHhwsA4tixY+LevXsiISFB7NmzR1SrVk1YWFiIxMREIYQQY8aMEQDEe++9p7H8qVOnBACxe/dujfIffvhBo/zu3bvCzMxMBAYGioKCAqndnDlzBAAxZswYqez48eMCgDh+/LgQQoi8vDzh7u4u6tatKx4+fKixnsJ9TZ48Wej6OCmPMeoSFxcnAGi8b0IIce7cOQFAvP3221JZce/n/v37BQCxaNEijfJBgwYJhUIh/vnnH6kMgAAgoqKipLIbN24Ic3Nz8corr0hlGRkZWmM9e/asACB27Nghlan3hdatW4ucnBypfMWKFQKAiIyMlMp8fX2Fr6+v1msPDw+XykJDQ7W2h5WVlc73Ub3uuLg4IYQQjx8/FnZ2duKNN97QaHf79m2hUqmk8ocPHwoAIiwsTKvPZ/Hy8hI1atQQ9+/fl8p+//13YWRkJEaPHi2VqffHr7/++pl9ql9Hp06dRF5enkZdSbdD0f1fiCfvd9F22dnZwsnJSbz66qsler1E+uJpTaIK1r17dzg6OsLV1RXDhg2DtbU19u3bh1q1amm0e+uttzSef/3111CpVOjRowdSUlKkR+vWrWFtbY3jx48DeHKUIicnB1OnTtU4+hIcHPzMsUVHRyMuLg7BwcFa1/2U5EjO8xhjYQMGDNB437y9veHj44ODBw9qtS36fh48eBDGxsaYNm2aRvmMGTMghMChQ4c0ytu3b4/WrVtLz+vUqYP+/fvj8OHDyM/PBwBYWFhI9bm5ubh//z7q168POzs7XLx4UWtMQUFB0pFE9RhNTEx0jr+8HD16FI8ePcJrr72msc2MjY3h4+MjbTMLCwuYmZnhxIkTWqdenyY5ORmXLl3C2LFj4eDgIJU3b94cPXr0MPi1vvHGG1rX0Om7HYqytrbGyJEjpedmZmbw9vbGv//+a9BYiYrD05pEFezjjz9Gw4YNYWJigpo1a6JRo0YwMtL8d5OJiYl0/YxabGwsUlNTUaNGDZ39qi+mvnHjBgCgQYMGGvWOjo6wt7d/6tjUp1jVp+n09TzGWFjR5QGgYcOG2Lt3r0aZrvfzxo0bcHFxgY2NjUZ5kyZNNMb4rHVlZGTg3r17cHJyQmZmJpYuXYrw8HDcunVL49q11NTUZ47f2toazs7Oz3UOstjYWAD/f+1jUba2tgAApVKJ5cuXY8aMGahZsybatWuHPn36YPTo0XByciq2f/X72KhRI626Jk2a4PDhwwbdoOHu7q5Vpu92KKp27dpa/xixt7fHH3/8UaoxEj0LwxlRBfP29pbu1iyOUqnUCmwFBQWoUaMGdu/erXMZR0fHMhtjacl1jLrez/IwdepUhIeHIzg4GO3bt4dKpYJCocCwYcNQUFBQ7usvDfW4du7cqTNkFb4LNDg4GH379sX+/ftx+PBhzJ07F0uXLsVPP/2Eli1bPrcxF1b4KJmaoduhuLtZRTE3ihAZiuGMqJKqV68ejh07ho4dO+r8QlJTT9oZGxuLl156SSq/d+/eM09HqS8a/+uvv9C9e/di2xV3ivN5jLEw9VGfwmJiYp56cXrhMRw7dgyPHz/WOHp27do1jTE+a12WlpZS6IyIiMCYMWOwatUqqU1WVhYePXpU7Pi7dOkiPU9PT0dycjJ69+79zPE/S0lvKFBv8xo1ajx1mxduP2PGDMyYMQOxsbHw8vLCqlWrsGvXLp3t1e/j33//rVV37do1VK9evcynNdF3OxBVNF5zRlRJDRkyBPn5+Vi4cKFWXV5envTF0717d5iammL9+vUa/9Jfu3btM9fRqlUruLu7Y+3atVpfZIX7Un+ZFm3zPMZY2P79+3Hr1i3p+fnz53Hu3LkS3VHXu3dv5OfnY8OGDRrla9asgUKh0Orj7NmzGtcrJSQkIDIyEj179pSOtBgbG2sdXVm/fr10TVpRW7duRW5urvR806ZNyMvLK5M7Aq2srEoURvz9/WFra4slS5ZojEXt3r17AJ5M+5GVlaVRV69ePdjY2CA7O7vY/p2dneHl5YXt27drjOevv/7CkSNHyiSIFqXvdiCqaDxyRlRJ+fr6YuLEiVi6dCkuXbqEnj17wtTUFLGxsfj666+xbt06DBo0CI6Ojpg5cyaWLl2KPn36oHfv3oiOjsahQ4dQvXr1p67DyMgImzZtQt++feHl5YVx48bB2dkZ165dw+XLl3H48GEAkC6MnzZtGvz9/WFsbIxhw4Y9lzEWVr9+fXTq1AlvvfUWsrOzsXbtWlSrVg2zZs165rJ9+/ZFly5d8P777yM+Ph4tWrTAkSNHEBkZieDgYK2pJzw9PeHv768xlQYAzJ8/X2rTp08f7Ny5EyqVCk2bNsXZs2dx7NgxjWlSCsvJyUG3bt0wZMgQ/P3339i4cSM6deqEfv36lfg9KE7r1q1x7NgxrF69Gi4uLnB3d4ePj49WO1tbW2zatAmjRo1Cq1atMGzYMDg6OuLmzZv4/vvv0bFjR2zYsAExMTHSWJs2bQoTExPs27cPd+7cwbBhw546lrCwMPTq1Qvt27fH+PHjpak0VCpVufzKhL7bgajCVdh9okQvOPWt/xcuXHhquzFjxggrK6ti67du3Spat24tLCwshI2NjWjWrJmYNWuWSEpKktrk5+eL+fPnC2dnZ2FhYSH8/PzEX3/9JerWrfvUqTTUTp8+LXr06CFsbGyElZWVaN68uVi/fr1Un5eXJ6ZOnSocHR2FQqHQmsahLMeoi3o6ibCwMLFq1Srh6uoqlEqlePnll8Xvv/9e4vfz8ePH4u233xYuLi7C1NRUNGjQQISFhWlM7yHEk6k0Jk+eLHbt2iUaNGgglEqlaNmypdb79vDhQzFu3DhRvXp1YW1tLfz9/cW1a9e0XpN6Xzh58qQICgoS9vb2wtraWowYMUJjugkhSj+VxrVr10Tnzp2FhYWFxvQkRafSUDt+/Ljw9/cXKpVKmJubi3r16omxY8dK04ekpKSIyZMni8aNGwsrKyuhUqmEj4+P2Lt3r873tqhjx46Jjh07CgsLC2Frayv69u0rrly5ojUG6DmVhq6/p5Juh+Km0vDw8NDqc8yYMaJu3boleq1E+lIIwSsaiahyi4+Ph7u7O8LCwjBz5syKHg4RkUF4zRkRERGRjDCcEREREckIwxkRERGRjPCaMyIiIiIZ4ZEzIiIiIhlhOCMiIiKSEYYzIiIiIhlhOCMiIiKSEYYzIiIiIhlhOCMiIiKSEYYzIiIiIhlhOCMiIiKSEYYzIiIiIhn5P6dAUnv7lqtXAAAAAElFTkSuQmCC"/>

**관찰**





- 우리는 위의 히스토그램이 매우 긍정적으로 치우쳐 있음을 알 수 있다.





- 첫 번째 열은 확률이 0.0에서 0.1 사이인 약 15000개의 관측치가 있음을 알려준다.





- 확률 > 0.5인 관측치가 적다.





- 그래서, 이 적은 수의 관측은 내일 비가 올 것이라고 예측한다.





- 관측의 대부분은 내일 비가 내리지 않을 것이라고 예측한다.


**임계값 낮추기**



```python
from sklearn.preprocessing import binarize

for i in range(1,5):
    
    cm1=0
    
    y_pred1 = logreg.predict_proba(X_test)[:,1]
    
    y_pred1 = y_pred1.reshape(-1,1)
    
    y_pred2 = binarize(y_pred1, threshold=i/10)
    
    y_pred2 = np.where(y_pred2 == 1, 'Yes', 'No')
    
    cm1 = confusion_matrix(y_test, y_pred2)
        
    print ('With',i/10,'threshold the Confusion Matrix is ','\n\n',cm1,'\n\n',
           
            'with',cm1[0,0]+cm1[1,1],'correct predictions, ', '\n\n', 
           
            cm1[0,1],'Type I errors( False Positives), ','\n\n',
           
            cm1[1,0],'Type II errors( False Negatives), ','\n\n',
           
           'Accuracy score: ', (accuracy_score(y_test, y_pred2)), '\n\n',
           
           'Sensitivity: ',cm1[1,1]/(float(cm1[1,1]+cm1[1,0])), '\n\n',
           
           'Specificity: ',cm1[0,0]/(float(cm1[0,0]+cm1[0,1])),'\n\n',
          
            '====================================================', '\n\n')
```

<pre>
With 0.1 threshold the Confusion Matrix is  

 [[13039  9687]
 [  570  5796]] 

 with 18835 correct predictions,  

 9687 Type I errors( False Positives),  

 570 Type II errors( False Negatives),  

 Accuracy score:  0.6474288464182593 

 Sensitivity:  0.9104618284637135 

 Specificity:  0.5737481298952741 

 ==================================================== 


With 0.2 threshold the Confusion Matrix is  

 [[17587  5139]
 [ 1354  5012]] 

 with 22599 correct predictions,  

 5139 Type I errors( False Positives),  

 1354 Type II errors( False Negatives),  

 Accuracy score:  0.7768114945689537 

 Sensitivity:  0.7873075714734528 

 Specificity:  0.7738713367948605 

 ==================================================== 


With 0.3 threshold the Confusion Matrix is  

 [[19673  3053]
 [ 2092  4274]] 

 with 23947 correct predictions,  

 3053 Type I errors( False Positives),  

 2092 Type II errors( False Negatives),  

 Accuracy score:  0.8231472569778633 

 Sensitivity:  0.6713792020106818 

 Specificity:  0.8656604769867112 

 ==================================================== 


With 0.4 threshold the Confusion Matrix is  

 [[20858  1868]
 [ 2732  3634]] 

 with 24492 correct predictions,  

 1868 Type I errors( False Positives),  

 2732 Type II errors( False Negatives),  

 Accuracy score:  0.841880929465145 

 Sensitivity:  0.5708451146716934 

 Specificity:  0.9178033969902315 

 ==================================================== 


</pre>
**코멘트**





- 이진법 문제에서는 예측 확률을 클래스 예측으로 변환하기 위해 기본적으로 임계값 0.5가 사용된다.





- 임계값을 조정하여 민감도 또는 특이도를 높일 수 있다.





- 민감도와 특이도는 반비례 관계이다. 하나를 늘리면 항상 다른 하나가 줄어들고 그 반대도 마찬가지이다.





- 임계값을 높이면 정확도가 높아지는 것을 볼 수 있다.





- 임계값 수준을 조정하는 것은 모델 작성 프로세스에서 수행하는 마지막 단계 중 하나여야 한다.


# **18. ROC - AUC** <a class="anchor" id="18"></a>



## ROC 곡선



분류 모델 성능을 시각적으로 측정하는 또 다른 도구는 **ROC 곡선**이다. ROC 곡선은 **수신기 작동 특성 곡선**을 나타낸다. **ROC 곡선**은 다양한 분류 모델의 성능을 보여주는 플롯이이다.

분류 임계값 수준.





**ROC 곡선**은 다양한 임계값 수준에서 **FPR(거짓 양성률)**에 대한 **TPR(참 양성률)**을 나타낸다.







**참양성률(TPR)**은 **재현율**이라고도 합니다. 'TP 대 (TP + FN)'의 비율로 정의된다.







**거짓양성률(FPR)**은 'FP 대 (FP + TN)'의 비율로 정의된된다.









ROC 곡선에서는 단일 지점의 TPR(참양성률) 및 FPR(거짓양성률)에 중점을 둘 것이다. 이는 다양한 임계값 수준에서 TPR 및 FPR로 구성된 ROC 곡선의 일반적인 성능을 제공한다. 따라서 ROC 곡선은 다양한 분류 임계값 수준에서 TPR 대 FPR을 표시한다. 임계값 수준을 낮추면 더 많은 항목이 양성으로 분류될 수 있다. True Positive(TP)와 False Positive(FP)를 모두 증가시킨킨다.



```python
# plot ROC Curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = 'Yes')

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()
```

<img src="data:image/png;base64,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"/>

ROC 곡선은 특정 상황에 대한 민감도와 특이도의 균형을 맞추는 임계값 수준을 선택하는 데 도움이 된다


**ROC-AUC**





**ROC AUC**는 **Receiver Operating Characteristic - Area Under Curve**를 나타낸다. 분류기 성능을 비교하는 기법이다. 이 기술에서는 '곡선 아래 면적(AUC)'을 측정합니다. 완벽한 분류기는 ROC AUC가 1인 반면 순수 무작위 분류기는 ROC AUC가 0.5이다.





따라서 **ROC AUC**는 곡선 아래에 있는 ROC 넓이의 백분율이다.



```python
# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))
```

<pre>
ROC AUC : 0.8624
</pre>
**코멘트**



- ROC AUC는 분류기 성능의 단일 숫자 요약이다. 값이 높을수록 더 나은 분류기이다.



- 우리 모델의 ROC AUC는 1에 근접한다. 따라서 분류기가 내일 비가 올지 여부를 잘 예측한다고 결론을 내릴 수 있다다.



```python
# calculate cross-validated ROC AUC 

from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(logreg, X_train, y_train, cv=5, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))
```

<pre>
Cross validated ROC AUC : 0.8660
</pre>
# **19. k-겹 교차 검증**<a class="anchor" id="19"></a>



```python
# Applying 5-Fold Cross Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg, X_train, y_train, cv = 5, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))
```

<pre>
Cross-validation scores:[0.84802784 0.84927387 0.84940277 0.8450565  0.84879474]
</pre>
평균을 계산하여 교차 검증 정확도를 요약할 수 있다.



```python
# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(scores.mean()))
```

<pre>
Average cross-validation score: 0.8481
</pre>
우리의 원래 모델 점수는 0.8449인 것으로 나타났다. 평균 교차 검증 점수는 0.8481이다. 따라서 교차 검증이 성능 향상으로 이어진다.


# **20. GridSearch CV를 사용한 하이퍼파라미터 최적화** <a class="anchor" id="20"></a>




```python
from sklearn.model_selection import GridSearchCV


parameters = [{'penalty':['l1','l2']}, 
              {'C':[1, 10, 100, 1000]}]



grid_search = GridSearchCV(estimator = logreg,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)


grid_search.fit(X_train, y_train)
```

<style>#sk-container-id-5 {color: black;background-color: white;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
             estimator=LogisticRegression(random_state=0, solver=&#x27;liblinear&#x27;),
             param_grid=[{&#x27;penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;]}, {&#x27;C&#x27;: [1, 10, 100, 1000]}],
             scoring=&#x27;accuracy&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5,
             estimator=LogisticRegression(random_state=0, solver=&#x27;liblinear&#x27;),
             param_grid=[{&#x27;penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;]}, {&#x27;C&#x27;: [1, 10, 100, 1000]}],
             scoring=&#x27;accuracy&#x27;)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(random_state=0, solver=&#x27;liblinear&#x27;)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(random_state=0, solver=&#x27;liblinear&#x27;)</pre></div></div></div></div></div></div></div></div></div></div>



```python
# examine the best model

# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))

# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))

# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))
```

<pre>
GridSearch CV best score : 0.8483


Parameters that give the best results : 

 {'penalty': 'l1'}


Estimator that was chosen by the search : 

 LogisticRegression(penalty='l1', random_state=0, solver='liblinear')
</pre>

```python
# calculate GridSearch CV score on test set

print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))
```

<pre>
GridSearch CV score on test set: 0.8488
</pre>
**comment**





- 우리의 원래 모델 테스트 정확도는 0.8456이고 GridSearch CV 정확도는 0.8488이다.





- GridSearch CV가 이 특정 모델의 성능을 향상시키는 것을 볼 수 있다.


# **21. 결과 및 결론** <a class="anchor" id="21"></a>



1. 로지스틱 회귀 모델 정확도 점수는 0.8456이다. 따라서 이 모델은 호주에 내일 비가 올지 여부를 예측하는 데 매우 효과적이다.



2. 내일 비가 올 것이라는 관측이 적다. 대부분의 관측은 내일 비가 내리지 않을 것이라고 예측한다.



3. 모델이 과적합의 징후를 보이지 않는다.



4. C 값을 높이면 테스트 세트 정확도가 높아지고 훈련 세트 정확도도 약간 높아진다. 따라서 더 복잡한 모델이 더 잘 수행되어야 한다는 결론을 내릴 수 있다.



5. 임계값 수준을 높이면 정확도가 높아진다.



6. 우리 모델의 ROC AUC는 1에 접근한다. 따라서 분류기가 내일 비가 올지 여부를 잘 예측한다고 결론을 내릴 수 있다.



7. 원래 모델 정확도 점수는 0.8456인 반면 RFECV 후 정확도 점수는 0.8500입니다. 따라서 거의 비슷한 정확도를 얻을 수 있지만 기능 세트는 줄었다.



8. 원래 모델에서 우리는 FP = 1175인 반면 FP1 = 1174입니다. 따라서 우리는 대략 같은 수의 잘못된 긍정을 얻는다. 또한 FN = 3087인 반면 FN1 = 3091입니다. 따라서 위음성이 약간 높아진다.



9. 우리의 원래 모델 점수는 0.8449인 것으로 나타났다. 평균 교차 검증 점수는 0.8481입니다. 따라서 교차 검증이 성능 향상으로 이어진다.



10. 원래 모델 테스트 정확도는 0.8456이고 GridSearch CV 정확도는 0.8507입니다. GridSearch CV가 이 특정 모델의 성능을 향상시키는 것을 볼 수 있다.


# **22. 참조** <a class="anchor" id="22"></a>









이 프로젝트에서 수행한 작업은 다음 책과 웹사이트에서 영감을 받음:





1. Hands on Machine Learning with Scikit-Learn and Tensorflow by Aurélién Géron



2. Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido



3. Udemy course – Machine Learning – A Z by Kirill Eremenko and Hadelin de Ponteves



4. Udemy course – Feature Engineering for Machine Learning by Soledad Galli



5. Udemy course – Feature Selection for Machine Learning by Soledad Galli



6. https://en.wikipedia.org/wiki/Logistic_regression



7. https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html



8. https://en.wikipedia.org/wiki/Sigmoid_function



9. https://www.statisticssolutions.com/assumptions-of-logistic-regression/



10. https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python



11. https://www.kaggle.com/neisha/heart-disease-prediction-using-logistic-regression



12. https://www.ritchieng.com/machine-learning-evaluate-classification-model/


