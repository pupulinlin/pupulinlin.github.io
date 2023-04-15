---
layout: single
title:  "jupyter notebook 변환하기!"
categories: coding
tag: [python, blog, jekyll]
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



## **2.1. 선형 방정식 구현**





로지스틱 회귀 알고리즘은 응답 값을 예측하기 위해 독립 또는 설명 변수가 있는 선형 방정식을 구현하여 작동한다. 예를 들어 공부한 시간과 시험에 합격할 확률의 예를 고려한다. 여기서 학습시간은 설명변수로 x1로 표시한다. 시험에 합격할 확률은 응답 또는 목표 변수이며 z로 표시된다.





하나의 설명 변수(x1)와 하나의 응답 변수(z)가 있는 경우 선형 방정식은 수학적으로 다음 방정식으로 주어집니다.



     z = β0 + β1x1



여기서 계수 β0 및 β1은 모델의 매개변수입니다.





설명 변수가 여러 개인 경우 위의 방정식은 다음과 같이 확장될 수 있다.



     z = β0 + β1x1+ β2x2+……..+ βnxn

    

여기서 계수 β0, β1, β2 및 βn은 모델의 매개변수이다.



따라서 예측 응답 값은 위의 방정식으로 주어지며 z로 표시된다.


## **2.2. sigmoid 함수**



z로 표시된 이 예측 응답 값은 0과 1 사이의 확률 값으로 변환된다. 예측 값을 확률 값에 매핑하기 위해 시그모이드 함수를 사용한다. 그런 다음 이 시그모이드 함수는 모든 실제 값을 0과 1 사이의 확률 값으로 매핑한다.



기계 학습에서 시그모이드 함수는 예측을 확률에 매핑하는 데 사용된다. 시그모이드 함수는 S자 모양의 곡선을 가진다. 시그모이드 곡선이라고도 한다.



Sigmoid 함수는 Logistic 함수의 특별한 경우이다. 다음 수학 공식으로 제공됩니다.



그래픽으로 시그모이드 함수를 다음 그래프로 나타낼 수 있다.


### Sigmoid Function



![Sigmoid Function](https://miro.medium.com/max/970/1*Xu7B5y9gp0iL5ooBj7LtWw.png)


## **2.3. 결정 경계**



시그모이드 함수는 0과 1 사이의 확률 값을 반환한다. 그런 다음 이 확률 값은 "0" 또는 "1"인 개별 클래스에 매핑된다. 이 확률 값을 불연속 클래스(합격/불합격, 예/아니오, 참/거짓)에 매핑하기 위해 임계값을 선택한다. 이 임계값을 결정 경계라고 합니다. 이 임계값 이상에서는 확률 값을 클래스 1에 매핑하고 그 아래에서는 값을 클래스 0에 매핑한다.



수학적으로 다음과 같이 표현할 수 있습니다.



p ≥ 0.5 => class = 1



p < 0.5 => class = 0



일반적으로 결정 경계는 0.5로 설정된다. 따라서 확률 값이 0.8(> 0.5)이면 이 관찰을 클래스 1에 매핑한다. 마찬가지로 확률 값이 0.2(< 0.5)이면 이 관찰을 클래스 0에 매핑한다. 이는 그래프에 표시된다.


![Decision boundary in sigmoid function](https://ml-cheatsheet.readthedocs.io/en/latest/_images/logistic_regression_sigmoid_w_threshold.png)


## **2.4. 예측하기**



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

# **7. Exploratory data analysis** <a class="anchor" id="7"></a>





[Table of Contents](#0.1)





Now, I will explore the data to gain insights about the data. 



```python
# view dimensions of dataset

df.shape
```

<pre>
(129804, 23)
</pre>
We can see that there are 142193 instances and 24 variables in the data set.



```python
# preview the dataset

df.head()
```

<pre>
         Date Location  MinTemp  MaxTemp  Rainfall  Evaporation  Sunshine  \
0  2008-12-01   Albury     13.4     22.9       0.6          NaN       NaN   
1  2008-12-02   Albury      7.4     25.1       0.0          NaN       NaN   
2  2008-12-03   Albury     12.9     25.7       0.0          NaN       NaN   
3  2008-12-04   Albury      9.2     28.0       0.0          NaN       NaN   
4  2008-12-05   Albury     17.5     32.3       1.0          NaN       NaN   

  WindGustDir  WindGustSpeed WindDir9am  ... Humidity9am  Humidity3pm  \
0           W           44.0          W  ...        71.0         22.0   
1         WNW           44.0        NNW  ...        44.0         25.0   
2         WSW           46.0          W  ...        38.0         30.0   
3          NE           24.0         SE  ...        45.0         16.0   
4           W           41.0        ENE  ...        82.0         33.0   

   Pressure9am  Pressure3pm  Cloud9am  Cloud3pm  Temp9am  Temp3pm  RainToday  \
0       1007.7       1007.1       8.0       NaN     16.9     21.8         No   
1       1010.6       1007.8       NaN       NaN     17.2     24.3         No   
2       1007.6       1008.7       NaN       2.0     21.0     23.2         No   
3       1017.6       1012.8       NaN       NaN     18.1     26.5         No   
4       1010.8       1006.0       7.0       8.0     17.8     29.7         No   

   RainTomorrow  
0            No  
1            No  
2            No  
3            No  
4            No  

[5 rows x 23 columns]
</pre>

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
### Drop  RISK_MM variable



It is given in the dataset description, that we should drop the `RISK_MM` feature variable from the dataset description. So, we 

should drop it as follows-



```python
df.drop(['RISK_MM'], axis=1, inplace=True)
```


```python
# view summary of dataset

df.info()
```

### Types of variables





In this section, I segregate the dataset into categorical and numerical variables. There are a mixture of categorical and numerical variables in the dataset. Categorical variables have data type object. Numerical variables have data type float64.





First of all, I will find categorical variables.



```python
# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
```


```python
# view the categorical variables

df[categorical].head()
```

### Summary of categorical variables





- There is a date variable. It is denoted by `Date` column.





- There are 6 categorical variables. These are given by `Location`, `WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday` and  `RainTomorrow`.





- There are two binary categorical variables - `RainToday` and  `RainTomorrow`.





- `RainTomorrow` is the target variable.


## Explore problems within categorical variables





First, I will explore the categorical variables.





### Missing values in categorical variables



```python
# check missing values in categorical variables

df[categorical].isnull().sum()
```


```python
# print categorical variables containing missing values

cat1 = [var for var in categorical if df[var].isnull().sum()!=0]

print(df[cat1].isnull().sum())
```

We can see that there are only 4 categorical variables in the dataset which contains missing values. These are `WindGustDir`, `WindDir9am`, `WindDir3pm` and `RainToday`.


### Frequency counts of categorical variables





Now, I will check the frequency counts of categorical variables.



```python
# view frequency of categorical variables

for var in categorical: 
    
    print(df[var].value_counts())
```


```python
# view frequency distribution of categorical variables

for var in categorical: 
    
    print(df[var].value_counts()/np.float(len(df)))
```

### Number of labels: cardinality





The number of labels within a categorical variable is known as **cardinality**. A high number of labels within a variable is known as **high cardinality**. High cardinality may pose some serious problems in the machine learning model. So, I will check for high cardinality.



```python
# check for cardinality in categorical variables

for var in categorical:
    
    print(var, ' contains ', len(df[var].unique()), ' labels')
```

We can see that there is a `Date` variable which needs to be preprocessed. I will do preprocessing in the following section.





All the other variables contain relatively smaller number of variables.


### Feature Engineering of Date Variable



```python
df['Date'].dtypes
```

We can see that the data type of `Date` variable is object. I will parse the date currently coded as object into datetime format.



```python
# parse the dates, currently coded as strings, into datetime format

df['Date'] = pd.to_datetime(df['Date'])
```


```python
# extract year from date

df['Year'] = df['Date'].dt.year

df['Year'].head()
```


```python
# extract month from date

df['Month'] = df['Date'].dt.month

df['Month'].head()
```


```python
# extract day from date

df['Day'] = df['Date'].dt.day

df['Day'].head()
```


```python
# again view the summary of dataset

df.info()
```

We can see that there are three additional columns created from `Date` variable. Now, I will drop the original `Date` variable from the dataset.



```python
# drop the original Date variable

df.drop('Date', axis=1, inplace = True)
```


```python
# preview the dataset again

df.head()
```

Now, we can see that the `Date` variable has been removed from the dataset.



### Explore Categorical Variables





Now, I will explore the categorical variables one by one. 



```python
# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)
```

We can see that there are 6 categorical variables in the dataset. The `Date` variable has been removed. First, I will check missing values in categorical variables.



```python
# check for missing values in categorical variables 

df[categorical].isnull().sum()
```

We can see that `WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday` variables contain missing values. I will explore these variables one by one.


### Explore `Location` variable



```python
# print number of labels in Location variable

print('Location contains', len(df.Location.unique()), 'labels')
```


```python
# check labels in location variable

df.Location.unique()
```


```python
# check frequency distribution of values in Location variable

df.Location.value_counts()
```


```python
# let's do One Hot Encoding of Location variable
# get k-1 dummy variables after One Hot Encoding 
# preview the dataset with head() method

pd.get_dummies(df.Location, drop_first=True).head()
```

### Explore `WindGustDir` variable



```python
# print number of labels in WindGustDir variable

print('WindGustDir contains', len(df['WindGustDir'].unique()), 'labels')
```


```python
# check labels in WindGustDir variable

df['WindGustDir'].unique()
```


```python
# check frequency distribution of values in WindGustDir variable

df.WindGustDir.value_counts()
```


```python
# let's do One Hot Encoding of WindGustDir variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()
```


```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).sum(axis=0)
```

We can see that there are 9330 missing values in WindGustDir variable.


### Explore `WindDir9am` variable



```python
# print number of labels in WindDir9am variable

print('WindDir9am contains', len(df['WindDir9am'].unique()), 'labels')
```


```python
# check labels in WindDir9am variable

df['WindDir9am'].unique()
```


```python
# check frequency distribution of values in WindDir9am variable

df['WindDir9am'].value_counts()
```


```python
# let's do One Hot Encoding of WindDir9am variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()
```


```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).sum(axis=0)
```

We can see that there are 10013 missing values in the `WindDir9am` variable.


### Explore `WindDir3pm` variable



```python
# print number of labels in WindDir3pm variable

print('WindDir3pm contains', len(df['WindDir3pm'].unique()), 'labels')
```


```python
# check labels in WindDir3pm variable

df['WindDir3pm'].unique()
```


```python
# check frequency distribution of values in WindDir3pm variable

df['WindDir3pm'].value_counts()
```


```python
# let's do One Hot Encoding of WindDir3pm variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()
```


```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).sum(axis=0)
```

There are 3778 missing values in the `WindDir3pm` variable.


### Explore `RainToday` variable



```python
# print number of labels in RainToday variable

print('RainToday contains', len(df['RainToday'].unique()), 'labels')
```


```python
# check labels in WindGustDir variable

df['RainToday'].unique()
```


```python
# check frequency distribution of values in WindGustDir variable

df.RainToday.value_counts()
```


```python
# let's do One Hot Encoding of RainToday variable
# get k-1 dummy variables after One Hot Encoding 
# also add an additional dummy variable to indicate there was missing data
# preview the dataset with head() method

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()
```


```python
# sum the number of 1s per boolean variable over the rows of the dataset
# it will tell us how many observations we have for each category

pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).sum(axis=0)
```

There are 1406 missing values in the `RainToday` variable.


### Explore Numerical Variables



```python
# find numerical variables

numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)
```


```python
# view the numerical variables

df[numerical].head()
```

### Summary of numerical variables





- There are 16 numerical variables. 





- These are given by `MinTemp`, `MaxTemp`, `Rainfall`, `Evaporation`, `Sunshine`, `WindGustSpeed`, `WindSpeed9am`, `WindSpeed3pm`, `Humidity9am`, `Humidity3pm`, `Pressure9am`, `Pressure3pm`, `Cloud9am`, `Cloud3pm`, `Temp9am` and `Temp3pm`.





- All of the numerical variables are of continuous type.


## Explore problems within numerical variables





Now, I will explore the numerical variables.





### Missing values in numerical variables



```python
# check missing values in numerical variables

df[numerical].isnull().sum()
```

We can see that all the 16 numerical variables contain missing values.


### Outliers in numerical variables



```python
# view summary statistics in numerical variables

print(round(df[numerical].describe()),2)
```

On closer inspection, we can see that the `Rainfall`, `Evaporation`, `WindSpeed9am` and `WindSpeed3pm` columns may contain outliers.





I will draw boxplots to visualise outliers in the above variables. 



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

The above boxplots confirm that there are lot of outliers in these variables.


### Check the distribution of variables





Now, I will plot the histograms to check distributions to find out if they are normal or skewed. If the variable follows normal distribution, then I will do `Extreme Value Analysis` otherwise if they are skewed, I will find IQR (Interquantile range).



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

We can see that all the four variables are skewed. So, I will use interquantile range to find outliers.



```python
# find outliers for Rainfall variable

IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

For `Rainfall`, the minimum and maximum values are 0.0 and 371.0. So, the outliers are values > 3.2.



```python
# find outliers for Evaporation variable

IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)
print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

For `Evaporation`, the minimum and maximum values are 0.0 and 145.0. So, the outliers are values > 21.8.



```python
# find outliers for WindSpeed9am variable

IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)
print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

For `WindSpeed9am`, the minimum and maximum values are 0.0 and 130.0. So, the outliers are values > 55.0.



```python
# find outliers for WindSpeed3pm variable

IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)
print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
```

For `WindSpeed3pm`, the minimum and maximum values are 0.0 and 87.0. So, the outliers are values > 57.0.


# **8. Declare feature vector and target variable** <a class="anchor" id="8"></a>





[Table of Contents](#0.1)



```python
X = df.drop(['RainTomorrow'], axis=1)

y = df['RainTomorrow']
```

# **9. Split data into separate training and test set** <a class="anchor" id="9"></a>





[Table of Contents](#0.1)



```python
# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```


```python
# check the shape of X_train and X_test

X_train.shape, X_test.shape
```

# **10. Feature Engineering** <a class="anchor" id="10"></a>





[Table of Contents](#0.1)





**Feature Engineering** is the process of transforming raw data into useful features that help us to understand our model better and increase its predictive power. I will carry out feature engineering on different types of variables.





First, I will display the categorical and numerical variables again separately.



```python
# check data types in X_train

X_train.dtypes
```


```python
# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical
```


```python
# display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical
```

### Engineering missing values in numerical variables






```python
# check missing values in numerical variables in X_train

X_train[numerical].isnull().sum()
```


```python
# check missing values in numerical variables in X_test

X_test[numerical].isnull().sum()
```


```python
# print percentage of missing values in the numerical variables in training set

for col in numerical:
    if X_train[col].isnull().mean()>0:
        print(col, round(X_train[col].isnull().mean(),4))
```

### Assumption





I assume that the data are missing completely at random (MCAR). There are two methods which can be used to impute missing values. One is mean or median imputation and other one is random sample imputation. When there are outliers in the dataset, we should use median imputation. So, I will use median imputation because median imputation is robust to outliers.





I will impute missing values with the appropriate statistical measures of the data, in this case median. Imputation should be done over the training set, and then propagated to the test set. It means that the statistical measures to be used to fill missing values both in train and test set, should be extracted from the train set only. This is to avoid overfitting.



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


```python
# check missing values in numerical variables in X_test

X_test[numerical].isnull().sum()
```

Now, we can see that there are no missing values in the numerical columns of training and test set.


### Engineering missing values in categorical variables



```python
# print percentage of missing values in the categorical variables in training set

X_train[categorical].isnull().mean()
```


```python
# print categorical variables with missing data

for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))
```


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


```python
# check missing values in categorical variables in X_test

X_test[categorical].isnull().sum()
```

As a final check, I will check for missing values in X_train and X_test.



```python
# check missing values in X_train

X_train.isnull().sum()
```


```python
# check missing values in X_test

X_test.isnull().sum()
```

We can see that there are no missing values in X_train and X_test.


### Engineering outliers in numerical variables





We have seen that the `Rainfall`, `Evaporation`, `WindSpeed9am` and `WindSpeed3pm` columns contain outliers. I will use top-coding approach to cap maximum values and remove outliers from the above variables.



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


```python
X_train.Evaporation.max(), X_test.Evaporation.max()
```


```python
X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()
```


```python
X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()
```


```python
X_train[numerical].describe()
```

We can now see that the outliers in `Rainfall`, `Evaporation`, `WindSpeed9am` and `WindSpeed3pm` columns are capped.


### Encode categorical variables



```python
categorical
```


```python
X_train[categorical].head()
```


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

We can see that two additional variables `RainToday_0` and `RainToday_1` are created from `RainToday` variable.



Now, I will create the `X_train` training set.



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

Similarly, I will create the `X_test` testing set.



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

We now have training and testing set ready for model building. Before that, we should map all the feature variables onto the same scale. It is called `feature scaling`. I will do it as follows.


# **11. Feature Scaling** <a class="anchor" id="11"></a>





[Table of Contents](#0.1)



```python
X_train.describe()
```


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

We now have `X_train` dataset ready to be fed into the Logistic Regression classifier. I will do it as follows.


# **12. Model training** <a class="anchor" id="12"></a>





[Table of Contents](#0.1)



```python
# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression


# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)


# fit the model
logreg.fit(X_train, y_train)
```

# **13. Predict results** <a class="anchor" id="13"></a>





[Table of Contents](#0.1)



```python
y_pred_test = logreg.predict(X_test)

y_pred_test
```

### predict_proba method





**predict_proba** method gives the probabilities for the target variable(0 and 1) in this case, in array form.



`0 is for probability of no rain` and `1 is for probability of rain.`



```python
# probability of getting output as 0 - no rain

logreg.predict_proba(X_test)[:,0]
```


```python
# probability of getting output as 1 - rain

logreg.predict_proba(X_test)[:,1]
```

# **14. Check accuracy score** <a class="anchor" id="14"></a>





[Table of Contents](#0.1)



```python
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))
```

Here, **y_test** are the true class labels and **y_pred_test** are the predicted class labels in the test-set.


### Compare the train-set and test-set accuracy





Now, I will compare the train-set and test-set accuracy to check for overfitting.



```python
y_pred_train = logreg.predict(X_train)

y_pred_train
```


```python
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
```

### Check for overfitting and underfitting



```python
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))
```

The training-set accuracy score is 0.8476 while the test-set accuracy to be 0.8501. These two values are quite comparable. So, there is no question of overfitting. 



In Logistic Regression, we use default value of C = 1. It provides good performance with approximately 85% accuracy on both the training and the test set. But the model performance on both the training and test set are very comparable. It is likely the case of underfitting. 



I will increase C and fit a more flexible model.



```python
# fit the Logsitic Regression model with C=100

# instantiate the model
logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)


# fit the model
logreg100.fit(X_train, y_train)
```


```python
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg100.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg100.score(X_test, y_test)))
```

We can see that, C=100 results in higher test set accuracy and also a slightly increased training set accuracy. So, we can conclude that a more complex model should perform better.


Now, I will investigate, what happens if we use more regularized model than the default value of C=1, by setting C=0.01.



```python
# fit the Logsitic Regression model with C=001

# instantiate the model
logreg001 = LogisticRegression(C=0.01, solver='liblinear', random_state=0)


# fit the model
logreg001.fit(X_train, y_train)
```


```python
# print the scores on training and test set

print('Training set score: {:.4f}'.format(logreg001.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg001.score(X_test, y_test)))
```

So, if we use more regularized model by setting C=0.01, then both the training and test set accuracy decrease relatiev to the default parameters.


### Compare model accuracy with null accuracy





So, the model accuracy is 0.8501. But, we cannot say that our model is very good based on the above accuracy. We must compare it with the **null accuracy**. Null accuracy is the accuracy that could be achieved by always predicting the most frequent class.



So, we should first check the class distribution in the test set. 



```python
# check class distribution in test set

y_test.value_counts()
```

We can see that the occurences of most frequent class is 22067. So, we can calculate null accuracy by dividing 22067 by total number of occurences.



```python
# check null accuracy score

null_accuracy = (22067/(22067+6372))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))
```

We can see that our model accuracy score is 0.8501 but null accuracy score is 0.7759. So, we can conclude that our Logistic Regression model is doing a very good job in predicting the class labels.


Now, based on the above analysis we can conclude that our classification model accuracy is very good. Our model is doing a very good job in terms of predicting the class labels.





But, it does not give the underlying distribution of values. Also, it does not tell anything about the type of errors our classifer is making. 





We have another tool called `Confusion matrix` that comes to our rescue.


# **15. Confusion matrix** <a class="anchor" id="15"></a>





[Table of Contents](#0.1)





A confusion matrix is a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.





Four types of outcomes are possible while evaluating a classification model performance. These four outcomes are described below:-





**True Positives (TP)** – True Positives occur when we predict an observation belongs to a certain class and the observation actually belongs to that class.





**True Negatives (TN)** – True Negatives occur when we predict an observation does not belong to a certain class and the observation actually does not belong to that class.





**False Positives (FP)** – False Positives occur when we predict an observation belongs to a    certain class but the observation actually does not belong to that class. This type of error is called **Type I error.**







**False Negatives (FN)** – False Negatives occur when we predict an observation does not belong to a certain class but the observation actually belongs to that class. This is a very serious error and it is called **Type II error.**







These four outcomes are summarized in a confusion matrix given below.




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

The confusion matrix shows `20892 + 3285 = 24177 correct predictions` and `3087 + 1175 = 4262 incorrect predictions`.





In this case, we have





- `True Positives` (Actual Positive:1 and Predict Positive:1) - 20892





- `True Negatives` (Actual Negative:0 and Predict Negative:0) - 3285





- `False Positives` (Actual Negative:0 but Predict Positive:1) - 1175 `(Type I error)`





- `False Negatives` (Actual Positive:1 but Predict Negative:0) - 3087 `(Type II error)`



```python
# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
```

# **16. Classification metrices** <a class="anchor" id="16"></a>





[Table of Contents](#0.1)


## Classification Report





**Classification report** is another way to evaluate the classification model performance. It displays the  **precision**, **recall**, **f1** and **support** scores for the model. I have described these terms in later.



We can print a classification report as follows:-



```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))
```

## Classification accuracy



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

## Classification error



```python
# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))
```

## Precision





**Precision** can be defined as the percentage of correctly predicted positive outcomes out of all the predicted positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true and false positives (TP + FP). 





So, **Precision** identifies the proportion of correctly predicted positive outcome. It is more concerned with the positive class than the negative class.







Mathematically, precision can be defined as the ratio of `TP to (TP + FP).`








```python
# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))
```

## Recall





Recall can be defined as the percentage of correctly predicted positive outcomes out of all the actual positive outcomes.

It can be given as the ratio of true positives (TP) to the sum of true positives and false negatives (TP + FN). **Recall** is also called **Sensitivity**.





**Recall** identifies the proportion of correctly predicted actual positives.





Mathematically, recall can be given as the ratio of `TP to (TP + FN).`










```python
recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))
```

## True Positive Rate





**True Positive Rate** is synonymous with **Recall**.




```python
true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
```

## False Positive Rate



```python
false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
```

## Specificity



```python
specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))
```

## f1-score





**f1-score** is the weighted harmonic mean of precision and recall. The best possible **f1-score** would be 1.0 and the worst 

would be 0.0.  **f1-score** is the harmonic mean of precision and recall. So, **f1-score** is always lower than accuracy measures as they embed precision and recall into their computation. The weighted average of `f1-score` should be used to 

compare classifier models, not global accuracy.





## Support





**Support** is the actual number of occurrences of the class in our dataset.


# **17. Adjusting the threshold level** <a class="anchor" id="17"></a>





[Table of Contents](#0.1)



```python
# print the first 10 predicted probabilities of two classes- 0 and 1

y_pred_prob = logreg.predict_proba(X_test)[0:10]

y_pred_prob
```

### Observations





- In each row, the numbers sum to 1.





- There are 2 columns which correspond to 2 classes - 0 and 1.



    - Class 0 - predicted probability that there is no rain tomorrow.    

    

    - Class 1 - predicted probability that there is rain tomorrow.

        

    

- Importance of predicted probabilities



    - We can rank the observations by probability of rain or no rain.





- predict_proba process



    - Predicts the probabilities    

    

    - Choose the class with the highest probability    

    

    

- Classification threshold level



    - There is a classification threshold level of 0.5.    

    

    - Class 1 - probability of rain is predicted if probability > 0.5.    

    

    - Class 0 - probability of no rain is predicted if probability < 0.5.    

    




```python
# store the probabilities in dataframe

y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - No rain tomorrow (0)', 'Prob of - Rain tomorrow (1)'])

y_pred_prob_df
```


```python
# print the first 10 predicted probabilities for class 1 - Probability of rain

logreg.predict_proba(X_test)[0:10, 1]
```


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

### Observations





- We can see that the above histogram is highly positive skewed.





- The first column tell us that there are approximately 15000 observations with probability between 0.0 and 0.1.





- There are small number of observations with probability > 0.5.





- So, these small number of observations predict that there will be rain tomorrow.





- Majority of observations predict that there will be no rain tomorrow.


### Lower the threshold



```python
from sklearn.preprocessing import binarize

for i in range(1,5):
    
    cm1=0
    
    y_pred1 = logreg.predict_proba(X_test)[:,1]
    
    y_pred1 = y_pred1.reshape(-1,1)
    
    y_pred2 = binarize(y_pred1, i/10)
    
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

### Comments





- In binary problems, the threshold of 0.5 is used by default to convert predicted probabilities into class predictions.





- Threshold can be adjusted to increase sensitivity or specificity. 





- Sensitivity and specificity have an inverse relationship. Increasing one would always decrease the other and vice versa.





- We can see that increasing the threshold level results in increased accuracy.





- Adjusting the threshold level should be one of the last step you do in the model-building process.


# **18. ROC - AUC** <a class="anchor" id="18"></a>





[Table of Contents](#0.1)







## ROC Curve





Another tool to measure the classification model performance visually is **ROC Curve**. ROC Curve stands for **Receiver Operating Characteristic Curve**. An **ROC Curve** is a plot which shows the performance of a classification model at various 

classification threshold levels. 







The **ROC Curve** plots the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)** at various threshold levels.







**True Positive Rate (TPR)** is also called **Recall**. It is defined as the ratio of `TP to (TP + FN).`







**False Positive Rate (FPR)** is defined as the ratio of `FP to (FP + TN).`









In the ROC Curve, we will focus on the TPR (True Positive Rate) and FPR (False Positive Rate) of a single point. This will give us the general performance of the ROC curve which consists of the TPR and FPR at various threshold levels. So, an ROC Curve plots TPR vs FPR at different classification threshold levels. If we lower the threshold levels, it may result in more items being classified as positve. It will increase both True Positives (TP) and False Positives (FP).






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

ROC curve help us to choose a threshold level that balances sensitivity and specificity for a particular context.


## ROC-AUC





**ROC AUC** stands for **Receiver Operating Characteristic - Area Under Curve**. It is a technique to compare classifier performance. In this technique, we measure the `area under the curve (AUC)`. A perfect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will have a ROC AUC equal to 0.5. 





So, **ROC AUC** is the percentage of the ROC plot that is underneath the curve.



```python
# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))
```

### Comments





- ROC AUC is a single number summary of classifier performance. The higher the value, the better the classifier.



- ROC AUC of our model approaches towards 1. So, we can conclude that our classifier does a good job in predicting whether it will rain tomorrow or not.



```python
# calculate cross-validated ROC AUC 

from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(logreg, X_train, y_train, cv=5, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))
```

# **19. k-Fold Cross Validation** <a class="anchor" id="19"></a>





[Table of Contents](#0.1)



```python
# Applying 5-Fold Cross Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg, X_train, y_train, cv = 5, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))
```

We can summarize the cross-validation accuracy by calculating its mean.



```python
# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(scores.mean()))
```

Our, original model score is found to be 0.8476. The average cross-validation score is 0.8474. So, we can conclude that cross-validation does not result in performance improvement.


# **20. Hyperparameter Optimization using GridSearch CV** <a class="anchor" id="20"></a>





[Table of Contents](#0.1)



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


```python
# examine the best model

# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))

# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))

# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))
```


```python
# calculate GridSearch CV score on test set

print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))
```

### Comments





- Our original model test accuracy is 0.8501 while GridSearch CV accuracy is 0.8507.





- We can see that GridSearch CV improve the performance for this particular model.


# **21. Results and conclusion** <a class="anchor" id="21"></a>





[Table of Contents](#0.1)


1.	The logistic regression model accuracy score is 0.8501. So, the model does a very good job in predicting whether or not it will rain tomorrow in Australia.



2.	Small number of observations predict that there will be rain tomorrow. Majority of observations predict that there will be no rain tomorrow.



3.	The model shows no signs of overfitting.



4.	Increasing the value of C results in higher test set accuracy and also a slightly increased training set accuracy. So, we can conclude that a more complex model should perform better.



5.	Increasing the threshold level results in increased accuracy.



6.	ROC AUC of our model approaches towards 1. So, we can conclude that our classifier does a good job in predicting whether it will rain tomorrow or not.



7.	Our original model accuracy score is 0.8501 whereas accuracy score after RFECV is 0.8500. So, we can obtain approximately similar accuracy but with reduced set of features.



8.	In the original model, we have FP = 1175 whereas FP1 = 1174. So, we get approximately same number of false positives. Also, FN = 3087 whereas FN1 = 3091. So, we get slighly higher false negatives.



9.	Our, original model score is found to be 0.8476. The average cross-validation score is 0.8474. So, we can conclude that cross-validation does not result in performance improvement.



10.	Our original model test accuracy is 0.8501 while GridSearch CV accuracy is 0.8507. We can see that GridSearch CV improve the performance for this particular model.



# **22. References** <a class="anchor" id="22"></a>





[Table of Contents](#0.1)







The work done in this project is inspired from following books and websites:-





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



So, now we will come to the end of this kernel.



I hope you find this kernel useful and enjoyable.



Your comments and feedback are most welcome.



Thank you



[Go to Top](#0)

