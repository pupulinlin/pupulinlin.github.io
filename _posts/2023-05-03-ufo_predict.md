---
layout: single
title:  "7차 과제: 머신러닝 웹앱 구현"
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


# 머신러닝 모델을 사용하여 Web App 만들기


머신러닝 모델을 통해 훈련된 모델을 'pickle'하고 Flask 앱에서 모델을 사용하여 Web App을 만든다.


# 0. 도구


작업에서는 Flask와 Pickle이 필요하며, 둘 다 Python에서 작동한다.



*   [Flask](https://palletsprojects.com/p/flask/)는 'micro-framework'이며 Python으로 웹 프레임워크의 기본적인 기능과 웹 페이지를 만드는 템플릿 엔진을 제공한다. [this Learn module](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott)을 통해서 Flask로 웹 페이지를 만드는 것을 연습할 수 있다. 



* [Pickle](https://docs.python.org/3/library/pickle.html)은 Python 객체 구조를 serializes와 deserializes하는 Python 모듈이다. 모델을 'pickle'하게 되면, 웹에서 쓰기 위해서 serialize 또는 flatten을 한다. 





# 1. 데이터 탐색 및 정재


[NUFORC](https://nuforc.org/)에서 모아둔, 8만여 개의 UFO 목격 데이터를 사용한다.


[ufos.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/3-Web-App/1-Web-App/data/ufos.csv)에는 목격된 `city`, `state` 와 `country`, 오브젝트의 `shape` 와 `latitude` 및 `longitude` 열이 포함되어 있다.


pandas, numpy를 import 하고 ufos 정보를 담은 csv 파일도 import 하고 데이터의 생김새를 살펴본다.



```python
import pandas as pd
import numpy as np

ufos = pd.read_csv('/content/ufos.csv')
ufos.head()
```


  <div id="df-a7527e58-ce0a-4359-8a2d-51f6be5e3150">
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
      <th>datetime</th>
      <th>city</th>
      <th>state</th>
      <th>country</th>
      <th>shape</th>
      <th>duration (seconds)</th>
      <th>duration (hours/min)</th>
      <th>comments</th>
      <th>date posted</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10/10/1949 20:30</td>
      <td>san marcos</td>
      <td>tx</td>
      <td>us</td>
      <td>cylinder</td>
      <td>2700.0</td>
      <td>45 minutes</td>
      <td>This event took place in early fall around 194...</td>
      <td>4/27/2004</td>
      <td>29.883056</td>
      <td>-97.941111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10/10/1949 21:00</td>
      <td>lackland afb</td>
      <td>tx</td>
      <td>NaN</td>
      <td>light</td>
      <td>7200.0</td>
      <td>1-2 hrs</td>
      <td>1949 Lackland AFB&amp;#44 TX.  Lights racing acros...</td>
      <td>12/16/2005</td>
      <td>29.384210</td>
      <td>-98.581082</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10/10/1955 17:00</td>
      <td>chester (uk/england)</td>
      <td>NaN</td>
      <td>gb</td>
      <td>circle</td>
      <td>20.0</td>
      <td>20 seconds</td>
      <td>Green/Orange circular disc over Chester&amp;#44 En...</td>
      <td>1/21/2008</td>
      <td>53.200000</td>
      <td>-2.916667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10/10/1956 21:00</td>
      <td>edna</td>
      <td>tx</td>
      <td>us</td>
      <td>circle</td>
      <td>20.0</td>
      <td>1/2 hour</td>
      <td>My older brother and twin sister were leaving ...</td>
      <td>1/17/2004</td>
      <td>28.978333</td>
      <td>-96.645833</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10/10/1960 20:00</td>
      <td>kaneohe</td>
      <td>hi</td>
      <td>us</td>
      <td>light</td>
      <td>900.0</td>
      <td>15 minutes</td>
      <td>AS a Marine 1st Lt. flying an FJ4B fighter/att...</td>
      <td>1/22/2004</td>
      <td>21.418056</td>
      <td>-157.803611</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a7527e58-ce0a-4359-8a2d-51f6be5e3150')"
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
          document.querySelector('#df-a7527e58-ce0a-4359-8a2d-51f6be5e3150 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a7527e58-ce0a-4359-8a2d-51f6be5e3150');
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
  


ufos 데이터를 새로운 제목의 작은 데이터프레임으로 변환하고 `Country` 특성이 유니크 값인지 확인한다.



```python
ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})

ufos.Country.unique()
```

<pre>
array(['us', nan, 'gb', 'ca', 'au', 'de'], dtype=object)
</pre>
지금부터 모든 null 값을 드랍하고 1-60초 사이 목격 정보만 가져와서 처리할 데이터의 수량을 줄인다.



```python
ufos.dropna(inplace=True)

ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]

ufos.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
Int64Index: 23285 entries, 2 to 72608
Data columns (total 4 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   Seconds    23285 non-null  float64
 1   Country    23285 non-null  object 
 2   Latitude   23285 non-null  float64
 3   Longitude  23285 non-null  float64
dtypes: float64(3), object(1)
memory usage: 909.6+ KB
</pre>
Scikit-learn의 LabelEncoder 라이브러리를 import해서 국가의 텍스트 값을 숫자로 변환:


LabelEncoder는 데이터를 알파벳 순서로 인코드한다.



```python
from sklearn.preprocessing import LabelEncoder

ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])

ufos.head()
```


  <div id="df-49785d39-e470-4b67-8160-379965a1f552">
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
      <th>Seconds</th>
      <th>Country</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>20.0</td>
      <td>3</td>
      <td>53.200000</td>
      <td>-2.916667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20.0</td>
      <td>4</td>
      <td>28.978333</td>
      <td>-96.645833</td>
    </tr>
    <tr>
      <th>14</th>
      <td>30.0</td>
      <td>4</td>
      <td>35.823889</td>
      <td>-80.253611</td>
    </tr>
    <tr>
      <th>23</th>
      <td>60.0</td>
      <td>4</td>
      <td>45.582778</td>
      <td>-122.352222</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3.0</td>
      <td>3</td>
      <td>51.783333</td>
      <td>-0.783333</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-49785d39-e470-4b67-8160-379965a1f552')"
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
          document.querySelector('#df-49785d39-e470-4b67-8160-379965a1f552 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-49785d39-e470-4b67-8160-379965a1f552');
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
  


# 2. 모델 만들기


이제 데이터를 훈련셋과 테스트셋으로 나누어 훈련할 준비를 한다.


`Seleceted_features`에 해당하는 특성인 `Second`, `Latitude`, `Longitude`를 통해서 `Country` 특성을 반환할 수 있도록 한다.



```python
from sklearn.model_selection import train_test_split

Selected_features = ['Seconds','Latitude','Longitude']

X = ufos[Selected_features]
y = ufos['Country']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

logistic regression을 사용해서 모델을 훈련:



```python
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print('Predicted labels: ', predictions)
print('Accuracy: ', accuracy_score(y_test, predictions))
```

<pre>
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        39
           1       0.79      0.23      0.35       239
           2       1.00      1.00      1.00         7
           3       1.00      1.00      1.00       118
           4       0.96      1.00      0.98      4254

    accuracy                           0.96      4657
   macro avg       0.95      0.84      0.87      4657
weighted avg       0.95      0.96      0.95      4657

Predicted labels:  [4 4 4 ... 4 4 4]
Accuracy:  0.9572686278720206
</pre>
<pre>
/usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
</pre>
`Country`와 `Latitude/Longitude`가 상관 관계에 있어서 대략적인 정확도가 95%로 나쁘지 않다.



만든 모델은 `Latitude` 와 `Longitude`에서 `Country`를 알 수 있어야 하므로 좋진 않지만, 정리한 원본 데이터에서 훈련을 해보고 웹 앱에서 모델을 쓰기에 좋은 사례이다.


# 3. 모델 'pickle' 하기 


모델을 pickle 하는 과정으로 코드 몇 줄로 할 수 있다. pickled되면, pickled 모델을 불러와서, 초, 위도와 경도 값이 포함된 샘플 데이터 배열을 대상으로 테스트한다.



```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

<pre>
[3]
</pre>
<pre>
/usr/local/lib/python3.10/dist-packages/sklearn/base.py:439: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names
  warnings.warn(
</pre>
모델은 영국 국가 코드인, '3'이 반환된다.


# 4. Flask 앱 만들기


이제 Flask 앱을 만들어서 모델을 부르고 비슷한 결과를 반환하게 해본다.



ufo-model.pkl 파일과 notebook.ipynb 파일 옆에 web-app 이라고 불리는 폴더를 만들면서 시작한다.



폴더에서 3가지 폴더를 만든다: static, 내부에 css 폴더가 있으며, templates도 있다. 지금부터 다음 파일과 디렉토리들이 있어야 한다.


```

web-app/

  static/

    css/

    templates/

notebook.ipynb

ufo-model.pkl

```


web-app 폴더에서 만들 첫 파일은 requirements.txt 파일이다. JavaScript 앱의 package.json 처럼, 앱에 필요한 의존성을 리스트한 파일이다. requirements.txt 에 해당 라인을 추가한다:


    

    scikit-learn

    pandas

    numpy

    flask

    


이제, web-app 으로 이동해서 파일을 실행한다:


<img src = "https://drive.google.com/uc?id=1p1tvLjVfsSUenAM7OEOwBic-hqv6YUVA" height= 40 width = 800 >


터미널에서 pip install을 타이핑해서, requirements.txt 에 나열된 라이브러리를 설치한다.


<img src = "https://drive.google.com/uc?id=1MTeMt5GD_yvZtlQKgZqTYz7J2XuRbml5" height=200 width=800>


이제, 앱을 완성하기 위해서 3가지 파일을 더 만든다:







1.   최상단에 app.py를 만든다.

2.   templates 디렉토리에 index.html을 만든다.

3.   static/css 디렉토리에 style.css를 만든다.





<img src = "https://drive.google.com/uc?id=10O_h6gH-Ju992RTW8zmB_2CZd2wMruz9" height = 300 width = 300>


style.css 파일


```

body {

	width: 100%;

	height: 100%;

	font-family: 'Helvetica';

	background: black;

	color: #fff;

	text-align: center;

	letter-spacing: 1.4px;

	font-size: 30px;

}



input {

	min-width: 150px;

}



.grid {

	width: 300px;

	border: 1px solid #2d2d2d;

	display: grid;

	justify-content: center;

	margin: 20px auto;

}



.box {

	color: #fff;

	background: #2d2d2d;

	padding: 12px;

	display: inline-block;

}

```


index.html 파일


```

<!DOCTYPE html>

<html>

<head>

  <meta charset="UTF-8">

  <title>🛸 UFO Appearance Prediction! 👽</title>

  <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}"> 

</head>



<body>

 <div class="grid">



  <div class="box">



  <p>According to the number of seconds, latitude and longitude, which country is likely to have reported seeing a UFO?</p>



    <form action="{{ url_for('predict')}}" method="post">

    	<input type="number" name="seconds" placeholder="Seconds" required="required" min="0" max="60" />

      <input type="text" name="latitude" placeholder="Latitude" required="required" />

		  <input type="text" name="longitude" placeholder="Longitude" required="required" />

      <button type="submit" class="btn">Predict country where the UFO is seen</button>

    </form>



  

   <p>{{ prediction_text }}</p>



 </div>

</div>



</body>

</html>

```


파일의 템플릿을 본다. 예측 텍스트: {{}}처럼, 앱에서 제공할 수 있는 변수 주위, 'mustache' 구문을 확인해보자. /predict 라우터에 예측을 보낼 폼도 있다.


마지막으로, 모델을 써서 예측으로 보여줄 python 파일을 만든다:<br>



app.py 파일


```

import numpy as np

from flask import Flask, request, render_template

import pickle



app = Flask(__name__)



model = pickle.load(open("./ufo-model.pkl", "rb"))





@app.route("/")

def home():

    return render_template("index.html")





@app.route("/predict", methods=["POST"])

def predict():



    int_features = [int(x) for x in request.form.values()]

    final_features = [np.array(int_features)]

    prediction = model.predict(final_features)



    output = prediction[0]



    countries = ["Australia", "Canada", "Germany", "UK", "US"]



    return render_template(

        "index.html", prediction_text="Likely country: {}".format(countries[output])

    )





if __name__ == "__main__":

    app.run(debug=True)

```


app.py 파일을 살펴보면



1.   먼저, 의존성을 불러오고 앱이 시작한다.

2.   그 다음, 모델을 가져온다.

3. 그 다음, index.html을 홈 라우터에 랜더링 한다.





/predict 라우터에서, 폼이 보내질 때의 과정





1.   폼 변수를 모아서 numpy 배열로 변환한다. 그러면 모델로 보내지고 예측이 반환된다.

2.   국가를 보여줄 때는 예상된 국가 코드에서 읽을 수 있는 텍스트로 다시 랜더링하고, 이 값을 템플릿에서 랜더링 할 수 있게 index.html로 보낸다.





이제 python app.py 또는 python3 app.py를 실행하면 - 웹 서버가 로컬에서 시작하고, 짧은 폼을 작성하면 UFOs가 목격된 장소에 대해 주목받을 질문의 답을 얻을 수 있다.


<img src = "https://drive.google.com/uc?id=1I_oH9W2-C0OfpESmyfeAGUpH-rUJU8hT" height =300 width = 600>

