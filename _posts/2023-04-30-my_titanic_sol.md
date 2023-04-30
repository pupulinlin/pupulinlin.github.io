---
layout: single
title:  "4차 과제에 대한 내용입니다."
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


# 1. 문제 정의하기


타이타닉호에 탑승했던 사람들의 정보를 바탕으로 생존자를 예측하는 문제이다.


# 2. 데이터 불러오기



```python
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_titanic_data():
    tarball_path = Path("datasets/titanic.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/titanic.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as titanic_tarball:
            titanic_tarball.extractall(path="datasets")
    return [pd.read_csv(Path("datasets/titanic") / filename)
            for filename in ("train.csv", "test.csv")]
```

데이터를 가져와서 로드해 본다.



```python
train, test = load_titanic_data()
```

적재한 훈련데이터를 확인하기 위해 head() 메서드를 이용하여 앞의 5열을 살펴본다.



```python
train.head()
```


  <div id="df-9bef7c2d-0ea7-473b-8d6f-59d1aa0e9698">
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9bef7c2d-0ea7-473b-8d6f-59d1aa0e9698')"
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
          document.querySelector('#df-9bef7c2d-0ea7-473b-8d6f-59d1aa0e9698 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9bef7c2d-0ea7-473b-8d6f-59d1aa0e9698');
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
  


# 3. 데이터 분석


각 특성의 의미를 간략하게 살펴보면



* Survivied는 생존 여부(0은 사망, 1은 생존; train 데이터에서만 제공),

* Pclass는 사회경제적 지위(1에 가까울 수록 높음),

* SipSp는 배우자나 형제 자매 명 수의 총 합,

* Parch는 부모 자식 명 수의 총 합을 나타낸다.



이제 각각 특성들의 의미를 알았으니, 주어진 데이터에서 대해 간략하게 살펴보자.



```python
print('train data shape: ', train.shape)
print('test data shape: ', test.shape)
print('----------[train infomation]----------')
print(train.info())
print('----------[test infomation]----------')
print(test.info())
```

<pre>
train data shape:  (891, 12)
test data shape:  (418, 11)
----------[train infomation]----------
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
None
----------[test infomation]----------
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 11 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  418 non-null    int64  
 1   Pclass       418 non-null    int64  
 2   Name         418 non-null    object 
 3   Sex          418 non-null    object 
 4   Age          332 non-null    float64
 5   SibSp        418 non-null    int64  
 6   Parch        418 non-null    int64  
 7   Ticket       418 non-null    object 
 8   Fare         417 non-null    float64
 9   Cabin        91 non-null     object 
 10  Embarked     418 non-null    object 
dtypes: float64(2), int64(4), object(5)
memory usage: 36.0+ KB
None
</pre>
범주형 특성과 수치형 특성들로 나뉨을 알 수 있다.


## 3.1. 범주형 특성에 대한 Pie chart


데이터 값의 분포를 보기 위한 `matplotlib` 와 `seaborn` 라이브러리를 불러온다.



```python
```


```python
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set() # setting seaborn default for plots
```

먼저 다음과 같은 범주형 특성의 분포를 보기 위해서 Pie chart를 만드는 함수를 정의해보자.



* Sex

* Pclass

* Embarked



```python
def pie_chart(feature):
    feature_ratio = train[feature].value_counts(sort=False)
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
 

    plt.plot(aspect='auto')
    plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')
    plt.title(feature + '\'s ratio in total')
    plt.show()

    for i, index in enumerate(feature_index):
        plt.subplot(1, feature_size + 1, i + 1, aspect='equal')
        plt.pie([survived[index], dead[index]], labels=['Survivied', 'Dead'], autopct='%1.1f%%')
        plt.title(str(index) + '\'s ratio')

    plt.show()
```

먼저 `Sex`에 대해서 Pie chart를 그려보면,



```python
pie_chart('Sex')
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYUAAAGbCAYAAAAr/4yjAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABC0klEQVR4nO3dd3wUZf4H8M/M7G6STTa9J4RASEINiHSQJsIpeiIqdj1QucNezvvhFQ/byXFnOfAUDzzBLqegiIgCKmIoUkKHQCiBBEhvm7LZnZnfHyEjIQFC2uzOft6vly9lsuW7S9zPzvN9nmcEVVVVEBERARD1LoCIiNwHQ4GIiDQMBSIi0jAUiIhIw1AgIiINQ4GIiDQMBSIi0jAUiIhIw1AgIiINQ4G83tixYzFz5ky3fTx3sHnzZqSmpmLz5s16l0LtjKHgBTIzM/HII49gzJgx6NOnD6644gpMnToV7733Xrs+b2pqKpYuXdquz9Fc27dvx7x581BeXq53KZds3bp1mDdvXqseY/78+VizZk0bVURGZtK7AGpf27dvx913343Y2FjcfPPNiIiIwKlTp7Bz5068++67uOuuu/QusUNkZGTg9ddfxw033IDAwMAGP1u1ahUEQWiz52rrx1u3bh0++OADPPzwwy1+jLfeegsTJkzAuHHj2qwuMiaGgsHNnz8fNpsNn376aaMPw6KiIp2qar2qqipYrdY2eSyLxdImj9Nej0fUkTh8ZHDHjx9Ht27dGgUCAISFhTU69sUXX2Dy5MlIS0vDoEGD8Pjjj+PUqVPazz/77DOkpqbi008/bXC/+fPnIzU1FevWrTtvLXa7HS+++CLGjh2L3r17Y+jQoZg6dSr27t17wdcwb948pKamIisrC08++SQGDhyI22+/HQBw4MABzJw5E1deeSX69OmD4cOH4+mnn0ZJSUmD+8+ZMwcAcOWVVyI1NRWpqanIyckB0HQP4MSJE3jkkUcwaNAg9O3bF1OmTMEPP/xwwTrrnft4S5cuRWpqKrZt24aXXnoJQ4YMQb9+/fDggw+iuLj4go81c+ZMfPDBBwCg1Z2amqr9vKqqCrNnz8aoUaPQu3dvTJgwAW+//TbO3vw4NTUVVVVVWLZsmXb/+vpyc3Mxa9YsTJgwAWlpaRg8eDAeeeQR7b0h78MzBYOLi4tDRkYGDh48iJSUlAve9s0338S//vUvXH311bjppptQXFyM999/H3fccQc+//xzBAYG4sYbb8Tq1asxe/ZsDB8+HDExMcjMzMTrr7+Om266CaNGjTrv4//1r3/FN998gzvvvBNJSUkoLS3Ftm3bcPjwYfTq1euir+XRRx9F586d8fjjj2sfehs2bMCJEycwefJkRERE4NChQ1iyZAmysrKwZMkSCIKAq666CseOHcOKFSvw9NNPIyQkBAAQGhra5PMUFhbi1ltvRXV1Ne666y6EhIRg2bJlmDFjBubOnYurrrrqorU25YUXXkBgYCAeeugh5ObmYvHixXjuuefw2muvnfc+t9xyC/Lz85Genq4FWz1VVTFjxgxs3rwZN910E3r06IH169djzpw5yMvLwx//+EcAwJw5c/DnP/8ZaWlpmDJlCgAgISEBALB7925kZGRg4sSJiI6ORm5uLj766CPcfffd+Oqrr+Dn59ei10oeTCVD++mnn9QePXqoPXr0UG+55RZ1zpw56vr169Xa2toGt8vJyVF79Oihvvnmmw2OZ2Zmqj179mxwPD8/Xx00aJA6depU1eFwqJMmTVJHjx6tVlRUXLCWyy+/XH322Wcv+TXMnTtXTUlJUZ944olGP6uurm50bMWKFWpKSoq6ZcsW7djChQvVlJQU9cSJE41uP2bMGPX//u//tD+/+OKLje5vt9vVsWPHqmPGjFFlWb5gvec+3meffaampKSov/nNb1RFUbTjf/vb39QePXqo5eXlF3y8Z599Vk1JSWl0fPXq1WpKSor6xhtvNDj+8MMPq6mpqWp2drZ2rF+/fg1qqtfU+5eRkaGmpKSoy5Yt045t2rRJTUlJUTdt2nTBWsnzcfjI4IYPH46PP/4YY8eOxYEDB7Bw4ULce++9GDlyJNauXavdbvXq1VAUBVdffTWKi4u1f8LDw9G5c+cGUxEjIiLwzDPPID09HXfccQf279+Pv/3tbwgICLhgLYGBgdi5cyfy8vJa9FpuvfXWRsd8fX21/3Y4HCguLkbfvn0B4KLDUuezbt06pKWlYcCAAdoxf39/3HLLLcjNzUVWVlaLHnfKlCkNGtADBgyALMvIzc1t0eP9+OOPkCSp0WSBadOmQVVV/Pjjjxd9jLPfP6fTiZKSEiQkJCAwMBD79u1rUV3k2Th85AXS0tLw+uuvo7a2FgcOHMCaNWuwaNEiPProo/j888/RrVs3HDt2DKqqYvz48U0+hsnU8Fdl4sSJWL58OX744QfccsstGDp06EXr+P3vf4+ZM2di9OjR6NWrF0aNGoVJkyahU6dOzXod8fHxjY6Vlpbi9ddfx8qVKxs1zisqKpr1uOc6efKkFixn69q1q/bziw3FNSU2NrbBn+v7PC2dJpubm4vIyMhGYZyUlKT9/GJqamrw1ltvYenSpcjLy2vQi2jp+0eejaHgRSwWC9LS0pCWlobExEQ8/fTTWLVqFR566CEoigJBELBgwQJIktTovufO9CkpKcGePXsAAFlZWVAUBaJ44RPPa665BgMGDMDq1auRnp6Ot99+GwsWLMC8efMu2Iuo5+Pj0+jYY489hoyMDNx7773o0aMHrFYrFEXBfffd1+ADzh2c7/3Rs87nn38eS5cuxT333IN+/frBZrNBEIQGfRvyLgwFL9W7d28AQH5+PoC6xqOqqoiPj0eXLl0uev/nnnsOlZWVePLJJ/Hyyy9j8eLFmDp16kXvFxkZiTvuuAN33HEHioqKcMMNN2D+/PnNCoVzlZWVYePGjXj44Yfx0EMPacePHTvW6LaXsm4gNjYWR48ebXT8yJEj2s870vlqj4uLw8aNG2G32xucLdTXGRcXd9HH/uabbzBp0qQGs6UcDgfPErwYewoGt2nTpia/8dVPHa0fEhk/fjwkScLrr7/e6PaqqjaY4rlq1SqsXLkSTz75JKZPn46JEyfitddea/KDtJ4sy40+aMLCwhAZGYna2toWvbamzmgAYPHixY2O1c+iac6H3ahRo7Br1y5kZGRox6qqqrBkyRLExcWhW7duLaq3peprP3eYaeTIkZBlWZuyWm/RokUQBAEjR47Ujlmt1iaHqZp6D9977z3IstwWpZMH4pmCwb3wwguorq7GVVddha5du8LpdGL79u34+uuvERcXh8mTJwOoO1N47LHH8PLLLyM3Nxfjxo2Dv78/cnJysGbNGkyZMgX33nsvioqKMGvWLAwePBh33nknAOAvf/kLNm/ejKeffhoffvhhk8MklZWVGDVqFCZMmIDu3bvDarViw4YN2L17d4v3CQoICMDAgQOxcOFCOJ1OREVFIT09vck59vVTXl999VVcc801MJvNGDNmTJML4KZPn46vvvoK999/P+666y4EBQXh888/R05ODubNm3fRYbK2Vl/7Cy+8gBEjRkCSJEycOBFjx47F4MGD8eqrryI3NxepqalIT0/H2rVrcc8992jTTusfY+PGjXjnnXcQGRmJ+Ph49O3bF6NHj8YXX3yBgIAAdOvWDTt27MCGDRsQHBzcoa+R3AdDweD+8Ic/YNWqVVi3bh0++eQTOJ1OxMbG4vbbb8eMGTMaLGqbPn06EhMTsWjRIvz73/8GAERHR2P48OEYO3YsAGDWrFmora3FSy+9pA1rhISE4LnnnsMDDzyAt99+G/fff3+jOnx9fXHbbbchPT0d3377LVRVRUJCAv76179qC9Fa4uWXX8bzzz+PDz/8EKqqYvjw4ViwYAGuuOKKBrdLS0vDo48+io8//hjr16+HoihYu3Ztk6EQHh6Ojz/+GP/4xz/w/vvvw+FwIDU1FfPnz8fo0aNbXGtLjR8/HnfddRe++uorLF++HKqqYuLEiRBFEW+++Sbmzp2LlStXYunSpYiLi8Mf/vAHTJs2rcFjzJw5E8888wxee+011NTU4IYbbkDfvn3xpz/9CaIo4ssvv4TD4UD//v3xzjvv4L777uvw10nuQVDZTSIiojPYUyAiIg1DgYiINAwFIiLSMBSIiEjDUCAiIg1DgYiINAwFIiLSMBSIiEjDUCAiIg1DgYiINAwFIiLSMBSIiEjDUCAiIg1DgYiINAwFIiLSMBSIiEjDUCAiIg1DgYiINAwFIiLSMBSIiEjDUCAiIg1DgYiINAwFIiLSMBSIiEjDUCAiIg1DgYiINAwFIiLSMBSIiEjDUCAiIg1DgYiINAwFIiLSMBSIiEjDUCAiIg1DgYiINAwFIiLSMBSIiEjDUCAiIg1DgYiINAwFIiLSMBQ8zNKlS5Gamori4mK9SyEiA2IoEBGRhqFAREQahkI7mDlzJq699lps2LAB1113HdLS0nDnnXciJycHpaWlePTRR9G/f3+MGzcOK1eu1O73ww8/YOrUqRg6dCj69++Pm2++GT/++ONFn6+2thavvPIKxowZg969e+Pqq6/Gl19+2Z4vkYgMyqR3AUZVUFCA2bNnY8aMGTCZTHjhhRfw+9//Hn5+fhgwYACmTJmCJUuW4KmnnkLfvn0RFxeHnJwcjBkzBtOmTYMoivjxxx8xffp0LF68GIMHDz7vcz366KPYvn07HnzwQSQlJWHdunV46qmnEBgYiFGjRnXgq3YPqqpCVlSoKiAKgCS17XcfRVGhqCoAwNTGj02kN4ZCOykrK8P777+P5ORkAEB+fj6ef/553H///XjwwQcBAH369MHq1auxZs0a3HPPPbjzzju1+yuKgsGDByMrKwtLliw5byhs2rQJ3333Hd5++22MGDECADB8+HAUFBRg3rx5hgwFWVGgqg0/kB21LlRUOVFqd6C4vAZldgfK7LUor6z7d5ndAXuVEy5Zgayodf+WVbiUun/LigKXrEJRVEiiAItZgsUswmKW4GOW4OdrgtXHDH+/un8HBlgQFuSL8GA/RIVYERroC1+fhv87uWQFggBIIoODPAdDoZ1ERkZqgQAAiYmJAIBhw4ZpxwIDAxEaGorTp08DAE6fPo1XX30VGzZsQEFBAdQz30Z79ep13udJT09HcHAwhgwZApfLpR0fNmwYZs2aBVmWIUlSW760DuOSFYiiAFEQANR98J8srMTxvAqcLLDjZEElcgvsOFlYicpqZ9s+eQsez8/HpAVFWJAfIkP80Dnahq7xwYgKsUIU616Hy6VAkgQIZ14XkTthKLSTwMDABn82m80AAJvN1uC4xWKBw+GAoiiYMWMGKioq8Mgjj6Bz587w8/PD3LlzcerUqfM+T0lJCUpLS88bHAUFBYiOjm7lq2l/sqIAqPtWraoqThZWYv/RYhw6UYLs0xXILbCjtMKhc5UXVu1wISffjpx8e6OfmU0iOkXZ0Dnahs4xgegSE4QusYEICfQFUBeAksigIP0xFNxEdnY29u3bh3//+98YN26cdrympuaC9wsKCkJoaCj+85//NPnz0NDQNq2zrbhkRRv+KbM7sP9YMTKzS3DweAmyckpRVeO6yCN4FqdLwZHcMhzJLWtwPMDPjO6JoejVNQxp3cKRFB8ESRQhywqEs86SiDoKQ8FNOBx134LrzygAIDc3FxkZGdrQU1OGDRuGhQsXwmw2o3v37u1dZospSt1QmCgKKKmowZZ9ecjIzMe+o8UoLr9w8BmZvdqJrfvzsHV/HgDAxywhJSEEvZLC0CcpDN07h8JiliCfGUrjmQS1N4aCm+jatSuio6Px8ssvQ1EUVFVVYe7cuYiMjLzg/YYPH44xY8bgvvvuw3333YfU1FRUV1cjKysL2dnZePHFFzvoFTRWfzbgkhXsPVKErfvzsP1APo7nVehWk7tzOGXsPlyI3YcL8TEASRTQLT4YA3pGYXhaLDpF2RoELFFbYyi4CYvFgnnz5uG5557Do48+ipiYGMyYMQObNm3Cnj17LnjfuXPn4j//+Q8++ugj5ObmwmazITk5GZMnT+6g6n9RHwQFJVXYtOc0th3Iw54jRXDUyh1eixHIiorM4yXIPF6CD1YdQFSoFYN7RWNoWgx6JoZBFAXIstLm027Jewlq/RQXohaqD4IyuwM/bMvBuowcHDpRqndZhhfgZ8aAHlEY0jsGA3pEwcciQVYUToGlVmEoUIvUB0G1w4X1O3KxLiMHe7IKofC3SRcWk4ghfWIwfnBnpHULh3Jm4R57EHSpGArUbKpat0pYVlRs3nMKP2zPwbYD+XDJit6l0Vkigv0wdkAnjB/SGZEh1gYzvYguhqFAF1U/Zl1YWo3l649gzc/ZqKhq48Vi1OYEAejVJQzjBiXgin5xMJtEqACnudIFMRTovOrDYMfBfCxffwTb9udxeMhD+fmYMOqyOEwe0w0x4QFsTtN5MRSoAUVRIQh1UyO/3ZSNlRuOIbeg8Qpd8kyCAAzsGY2bxnRDjy5hHFqiRhgKBKAuDERRQEFJFT79Pgvfbz2BaoexVhVTQykJIZg8uhuG9omBqqo8cyAADAWvVx8Gpwor8dG3mfgxIwcyx4i8SnSYFb8emYQJgzvDJIlcFOflGApeqj4Mcgvs+GDVAaTvzGW/wMvZrGZMHpOM60d2hSgIPHPwUgwFL6OoKkRBQF5RJd5bdQDrM3IYBtRAaKAvbp+QiqsGdYaiquw5eBmGghdRFBWlFQ68t2o/vtt6QttDh6gpcREBuPuaHhiWFsvZSl6EoeAFZFmBoqr439pDWPp9FhxO7kNEzZfcKRi/ubYn0rpFcBsNL8BQMLD6/4HTd+bi7S/3oqCkWu+SyIP1TY7A/df3RkJ03YWiuIWGMTEUDKj+r/R4XgXmL92FPYeLdK6IjEIUBVwzLBF3X9MTFpPIISUDYigYjKwoqHHIWPzVPnyzOZt9A2oXwTYfTLuuF8Zc3olDSgbDUDCI+v8xv9l0DItW7IO9rS9kT9SEPknheOSWfogMsXJ9g0EwFAxAVhSUVdTitY+3I+Nggd7lkJcxm0RMGZeCm69MhqqCU1g9HEPBg9UvQFu9ORsLl+8x3MXuybMkRNvwxG390SUuiDuxejCGgoeSFQXl9lr865MMbDuQr3c5RADqril9+4TuuPnKZCiqyl6DB2IoeJj6s4O1W45jwee7UcmzA3JDvbqG4ak7L0ewzYfB4GEYCh5ElhVU1rjw2kfbsWV/nt7lEF2Qv68JD9zUFyMvi9e2VyH3x1DwEIqqIjO7BLMXb0FxeY3e5RA12+j+8Xjw5r4wSSKb0B6AoeDm6r9hLf3+EN5duZ/bWpNHigq14vd3XI6UziE8Y3BzDAU3JssKal0KXvlwOzbtOaV3OUStIooCpl3XC9ePTIKqqtwmw00xFNyUoqg4kVeBF9/5GaeKKvUuh6jNXDmwEx66uR8EAWxCuyGGgpup/wa1enM25i/dhVqXondJRG0uJSEEf5k2GAFWM/sMboah4Ebq9yl647Od+GZTts7VELWv0EBf/GnqIHSLD+YWGW6EoeAmZEWBy6XgxUU/IyOTW1WQdzBJIh64MQ1XDe6sdyl0BkPBDciygoqqWjzzn404erJc73KIOtzE4V0wfVIfQABnJ+mMoaAzWVFwsqASz/xnAwpLuf6AvNegXtGYefdAiCIb0HpiKOhIUVTsOVKIF9/5mZvZEQHo3TUMz9w3hBfw0RFDQUdrtxzH6//bAZfMvwKiel3jgvD8b4fB39fEYNABQ0Enn6zOxPurDuhdBpFbign3x4szhiPU5sNg6GAMBR188M0BfPxtpt5lELm10EBfvPC7YYgN92cwdCCGQgd7/+v9+GTNQb3LIPIIAX5mzLp/KLp1CobEtQwdgqHQgd5buR9L1jIQiC6Fj0XCc9OHIrVzCGcldQC+wx1k8Vf7GAhELeColTFrwSYcPVkOWea2L+2NodABFq3Yi0+/O6R3GUQeq9rhwl/mb0BOgZ3B0M4YCu3sv1/uxWffZ+ldBpHHs1c78ac305FXUsVgaEcMhXb04TcHsOwHBgJRWymz1+Lpf6ejuLwGLgZDu2AotANFVbF6czY+4rRTojZXXF6Dp99IR3llLc8Y2gFDoY3JioqMA/n496c79S6FyLDyiqvwxzfSUVnjgqwwGNoSQ6ENybKCYyfLMPvdLbyWMlE7yy2wY9aCjVCUurNzahsMhTYiywqKymvw1wUbUVMr610OkVc4dKIUL3+wjdtttyGGQhuQFQXVDhf+PH8Dyuy1epdD5FXSd53Ee1/v17sMw2AotJKiqJBlFX9dsAmnCiv1LofIKy1ZcxA/bM/RLmlLLcdQaCVRFPDqx9tx8HiJ3qUQebW5n2Qg60QpZyS1EkOhFRRVxefrsvDTjpN6l0Lk9ZwuBc//dzNKKhwMhlZgKLSQLCvIzC7GohX79C6FiM4otTswa8FGuGSFQ0ktxFBoAVlRYa924qVFnHpK5G6yT1fglQ+3Q+RW2y3CUGgBAcDfFv2MkgqH3qUQURM27D6FlRuO8myhBRgKLfD28j3Yd7RY7zKI6AIWfrEHJ/IruEfSJWIoXAJZUbF+Ry6Wrz+idylEdBFOl4LZi7dAUVTwWmLNx1BoJllRkF9chbmfZOhdChE1U06+Hf/+dCcErnhuNpPeBXiSOe9tNfwWFjVlOSg6uBrVxcegyi6Y/UMRlDAYIV1GNLqt7KzGse/nQK6tREz/O2GLTbvo48vOahQf+g7203vgqimD5BMAa3gywlLGwewXot2uuvgY8vd8jtrKQvgGxSMqbTIsAZENHit/zxeorSxA/OD7Wv/CybC+23oC/VIiMPKyeF7nuRl4ptAMqqrio28ykZVTqncp7aqy4CBOpP8bsqMSYclXIqLXr+Ef2QOu6rImb1+U+S0U2dnsx1dVBTmbFqI0eyMConsjsvf1sMX2g/3ULpxIfwOKqwZAXXDkblkEk28gInpOhKq4cHLre1DVX8aGHRWnUXZ8MyJ6Xte6F01e4Y1PdyK/mBfnaQ6eKVyELCs4nFuG/xn8cpqyswand3wM/8geiLn8TgjChb8vOMpPozR7I8KSx6Ho4LfNeo6akuNwlJ1AZO9JCE4cph23BEQgb+f/UFmQBVtMb9SUHIeqOBFz+V0QJTP8I1Jx9LvZcFYWamcLBXu/RFDCYPjYolr+oslr1NTKeGnxz3j1sVF6l+L2eKZwAaqqQlZU/PP9bYaf2lZxMgOyw46w1AkQBBGKq7bBN/Nz5e9djoDo3vAL7dLs51BcdVN4JUtAg+Mmn0AAgCiZ624nOyGIZu3PotmqHQcA++k9qCnLRVjKVc1+bqKjJ8uxZO1BbrN9ETxTuABBELBoxT6cKjL+RndVBVkQTb5w1ZTj5NbFcFYWQpAsCIzvj4ie12kf0ABQcXIXakqOIXH07+Gsav6eT77B8RAkC4oOfgvJYoXZPwLOqkIU7P8KPkGdYA3vVne7oFgorhoUH14HW0waSo6uh2jyhSUgAorsQsG+FQhPHQ/JYm3z94GMbcmaQxh5WTyiw6yQRH4nbgpD4TxkWUHm8RKsSPeO6ae1lYVQVRknty5CUKdB8Ot+NaqLjqD0WDoUZzVi+t8BoO7besH+FQjpegXM1tBLCgXJ4o+Y/ncgb9enyNn0H+24NSIFsZffBUGUAABmayjCu1+NwgNfo3D/VxBEM6L63gRRsqDo0HcQJAuCOg9p2zeAvIJLVvDqR9vxj4ev0LsUt8VQOA9ZUfHaRxnwljNNVXZAlZ0I6jwEkb2vBwDYYvpAVVwoO74ZYSnjYQmIQHHW91AVGaHdxrboeSSLP3yD4uAbkggfWxQc5SdRfPgHnN65BLGX36XdLjRpFALj+8NZVQyLfwQkixWumjIUZ32HuIH3QFUUFOxfDvvpfZB8bIjsdR38QhPb4q0gg8vMLsGKn47imuFdOBupCTx/aoKqqnh/1QGvGDaqJ5wZHrLF9mtw3BZ3GQCgpvQ4nFXFKDm8DuHdfwXR5HPJz1FbWYScTW8hsNNAhCWPRUB0L4SlXIXI3jfAfmo3KvMPNLi9yccGv5DO2jBRwf6vYQ1PhjU8GcWH1qCqMAsxl9+BgOheyP35v5Cd1S145eSN3vt6P8oqHIbvFbYEQ+EcsqLgVGElvlx/WO9SOlR9s9fkc24TuO7Pcm01CjO/hck3ENawJDiriuGsKobLUXHm55VwVhVfsDldnrMVquyCf2SPBscDonoCqFubcD7VJdmwn9qFiJ7XAgAqTu5ASNJo+IV0RljyWIhmX1Tm8epb1DzVDhfmLsngpnlN4PDROSRRxJtLd8Ele9c3CJ+geFQVHoKrprzBIjFXTTkAQPLxh6u6FM6qIhz9bnaj++fvWQYASJrwLCSzX5PPITvsZ/6rYXDUB8n5AkVVVeTvWY7gLiNg8Q/T6jL5Bmq3MfkEwlXT9HoKoqZsO5CPH7bn4Iq+sZAkfj+ux1A4i0tWsGXfaew4WKB3KR3OFpuGksPfo+z4z9osIAAoO/4zIIiwhnWF2S8Ycm3DITVHRR6KMr858609AaJkAQAoci1c1aWQLP6QLP4AALN/OAAVFSd3IajTAO0xKnJ3AKibddSU8pytcNWUIiz5lz6G5GNDrT0f/hEpUBUZzqoiSD62tngryIss+Hw3BvaMglUUuBXGGQyFs6iqigVf7NG7DF34BsUhsNNAlJ/YAlVVYA3riqqiI7Cf2oXQbmNg8g2CyTeo0f1Ek9+Z+8cjILq3drym5ARyNr2F0ORxCE8dDwAI6jQAJUd+RP7uz+Aoy4XFFgVHWS7KTmyBxRbV4P71FFcNCg+sOtPH8NWO22L6oOjgGkBVUV1yDIrshH9k97Z+W8jgyitr8f7X+zF9Uh+9S3EbDIUzFEXFkjUHUVDivc3KqD6TYfYLRtmJrbCf3guzXzAiel6HkK5tM31PsvgjYcQjKDr4LSrz96Ps+CaIZiuCOg1AePerIYiNfx2LDq6FyTcIgfEDGhwPSxkPubYSRYfWwORjQ+zldzXqhxA1x9cbjuHaEV0RHebP2UgABJV7ykJRVBSVVeN3s9ei1sW9UYi8Tf/USDw7fajeZbgFdlcAiKKA+ct2MxCIvNT2zHxsP5DHC/KAoQBZVrA7qxA/7z2tdylEpKP/frmXw0dgKECSRLz79T69yyAinWWfrsB3W094/fbaXh0Ksqxge2Y+Dhxr/v49RGRc76/aD29f5OzVoSBJIt7/mqtgiahOYWkNlq8/DNmLk8FrQ6F+odqhE6V6l0JEbmTZD1lePYTktaFgkkS8v+rAxW9IRF6lzF6LVRuPeW0weGUoyLKCjbtP4kgu98ohosaWrcsCvHQikleGgigK+IBnCUR0HoWlNfhui3fORPK6UHDJCjbsOons0xV6l0JEbuzT7w555dbaXhcKJknEsh+861oJRHTpThZWIn3nSa9b5exVoSArCrJOlCLzONclENHFLVl7ECYvu9aCV71aSRTx+Y88SyCi5jl6shzb9ud5VW/Bq0KhzO5A+s5cvcsgIg+y9Icsr7oym9e8UkVRseKnI153mU0iap3dhwuRV1wJb7nKgPeEgqpi1cZsvcsgIg+jqsBX6cfgJZngHaHgkhWs256DUrtD71KIyAN9t/U4zxSMxCSJ+HL9Eb3LICIPVWavxYbdp7xieqrhQ0FRVRzOKcVhbmlBRK3w9YZjXjE91fivUAXWbDmudxVE5OF2Hy7E6SLjN5yNHwoA1u/gNFQiar2VG44avuFs6FCQZQU7DuajzF6rdylEZABrt5zgmYInkyQR3209oXcZRGQQ5ZW1yDhYAFkxbsPZ0KHgcMrYtPe03mUQkYGs35ELUTDu7qmGDYX6LbIdtbLepRCRgWzeexqKgYeQDBsKJknE99s4dEREbauy2omdhwoNO4Rk2FAor3Rg56FCvcsgIgP6ycBDSIYMhbqho1NQFOOe4hGRfjbtOWXYqamGDAWTJGLr/jy9yyAig6qocmL3YWMOIRkyFGRZwa4sDh0RUfsx6iwkw4WCoqjYe7QI1Q6X3qUQkYFt2nNK7xLaheFCAQB+3suhIyJqX2X2Whw9WW64Fc6GCwVRFLDtAEOBiNpfxsF8yAab0GK4UCgoqUJOvl3vMojIC+w8VGC47bQN9WpcsoKf9/EsgYg6xr6jxZANduEdQ4WCSRKxjVNRiaiDOGplHDxeYqi+gqFCQVVV7DlSpHcZRORFth8sMNRCWUOFQm6+nVNRiahD7TpUAMlAfQXDvBKXrPAsgYg63MHjJah1Gmc3ZsOEgiQKyDxeoncZRORlXLKKfUeLDDOEZJhQEAQBB44V610GEXmhA9klhrnGgmFCoarGidwCrk8goo53OKfMMOsVDPEqFEVFZnaJYbeyJSL3diS3VO8S2owhQkFVVezj0BER6SS/pBpVNU69y2gThggFSRKRmc1QICL9HM4pM8QiNkOEAgAczS3XuwQi8mJZOaWQZYaCW6iqcaLU7tC7DCLyYodzSmEyef5Hque/AoC7ohKR7g7nluldQpvw+FBwyQqOn67Quwwi8nInC+yGWNns8aEgCEBOPkOBiPSlqMDpokq9y2g1jw8FSRQ5fEREbuFkYaXHb3fh8aEAgCuZicgt5BVVefzlOT0+FGRFwalCzz9lIyLPl1dcBUkU9C6jVTw+FApKqj0+mYnIGE4XV0JkKOhHVVWcyGOTmYjcQ15xld4ltJpHh4KsqCgqq9G7DCIiAEA+Q0F/xeUMBSJyDzW1MuxVtXqX0SoeHQqSKDAUiMitePoQkkeHgiAwFIjIveSXVHv0bqkeHQoAUFrBjfCIyH1UVjs9ekakx4dCeaVnj98RkbHYq2s9+iqQDAUiojZUWe2E4MFLFTw6FJwuBdUOl95lEBFpKqtdED04FTw6FCoNck1UIjIOe3WtR69q9uhQqK31/L3LichY7NWe/WXVo0PBKSt6l0BE1EAlQ0E/ThdDgYjcC88UdOR0cfiIiNxLdY1nT37x6FCodfJMgYjci8vDh7U9OhQ4fERE7kbx5JVr8PBQqOXwERG5GQ/PBJj0LqClFEWFk8NH1EYeuKkv4iL89S6DDEASPfq7tueGggrV48fuyD3c+avuuHpoIuSqco/e3ZLcg+DBq5kBDw4FAIBnv/fkBronhuDmsd1QeWgL8pbM1rscMgDRLwCJTyzWu4wW89jzHFEQ4GOW9C6DPJivRcRz9w+BUlmKguWv610OGYXgsR+rADw4FASGArXS3x4YAV+LhLzP/gGlxq53OWQQgof3FDy6ej8fzx79Iv3c+avuSO4UguK178Jx8pDe5ZCR8ExBP74WninQpTu7j1D28wq9yyGDESSz3iW0ikeHgoWhQJeIfQRqb6KvZ09t9uhQYE+BLhX7CNTeJL8AvUtoFY8OBQtDgS4B+wjUEURfhoJuLCaGAjUP+wjUUUS/AI9eBOnZoWAW4cFXvaMOwj4CdSTJNwBQPHdfNo8OBUEQEBTgo3cZ5ObYR6COJPoFAOCZgm5CA331LoHcGPsI1NHqGs2eO4TBUCDDYh+B9CD62gDRc/udHh0KqqoiNIihQI2xj0B6kayBHr1TqkeHgiyrPFOgJrGPQHqR/IP0LqFVPDoUIHD4iBpjH4H0I8AUGK53Ea3i0aEgiQLCOHxEZ2EfgfQk2UIgmLj3kW4EQUBEiFXvMshNsI9AejOHROtdQqt5dCgA4JkCadhHIL2ZQ2I8ejUzYIBQsFkt8PfldRW83R0TUtlHIN2ZQqI8ejUzYIBQAIBO0Ta9SyAdpSaEYMqVyewjkO7MITG8yI7eVFVFQlSg3mWQTiwmEc//ln0Ecg/m8DhejlNvsqwiIYpnCt7qpQfZRyD3YQ6O0ruEVvP4UJAkAZ1jGAre6I4JqUhJYB+B3INoDYRo8fyJLx4fCoIgIDGGw0fehn0Ecjfm0Bi9S2gTHh8KABBs84WVM5C8BvsI5I58ortCVRW9y2g1Q4QCAHSK5BCSt2AfgdyRT2wy4OFrFACDhIKqqkjgtFSvwD4CuSvf+O4QPHjL7HqGCAVZUZHcKVjvMqidsY9A7kr09Yc5xPNnHgEGCQWTJKJPkmfvTEgXxj4CuTOfmCS9S2gzhggFAIiPssHfz7N3J6TzYx+B3JlPbDJUD9/eop5hQgEAuncO0bsEagfsI5C784lNhidfl/lshgkFl6ygZ5cwvcugNsY+AnkC3/hUj9/eop4xXgXqLrjTO4mhYCTsI5AnkGxhkKzGWUBrmFAQBAHJnYJhkoxxCkfsI5Bn8I1L0buENmWYUAAAs0lCUlyw3mVQG2AfgTyFX1I/qLJL7zLajKFCQVFU9OgSqncZ1ErsI5AnsXYbAEEyzjY7hgoFFSr6p0bqXQa1AvsI5EnMEQkwBQTrXUabMlQoSKKIPt3C4Wvx/KXm3op9BPIk1qTLoCqevwne2QwVCkDd6uZ+KTxb8ETsI5CnsSYPMMryBI3hQsElKxjUyxh7kHgT9hHI0wgW37r1CR5+TeZzGevVoO5MYUivGAgGS28jYx+BPJFfYpohdkU9l+FCAQBs/hYkd+KWF56CfQTyRNakyww1FbWeIUNBlhUM6skhJE/APgJ5Kmuysaai1jNkKIiigCF9jHG9VCNjH4E8lSWyM0w2Y66JMmQoCIKAztGBiAj207sUOg/2EciTBfQeCVU2xlbZ5zJkKAB1q5tH9IvVuww6D/YRyGMJIgLSRkOQjNdkBgwcChCAqwZ11rsKagL7COTJfDv3gsk/WO8y2o1hQ0EUBHSKsiEpLkjvUugs7COQp7P1HmnIWUf1DBsKQN1CtrEDOuldBp3BPgJ5OsFkgX+PYYacdVTP0KFgkkSMHdCJ11hwE+wjkKezJg+AaPHVu4x2ZehQAIAAqwWXd+eaBb2xj0BGYEsbA1Ux5qyjeoYPBVlWMG5ggt5leDX2EcgIRGsg/Lr2M+TWFmczfChIkoiBPaNgs5r1LsUrsY9ARhHQc7jhdkRtiuFDAQAEUcCo/vF6l+GV2EcgowgccLXeJXQIrwgFALh2RFe9S/A67COQUfh17QdLWJzhtsluivFfIerWLMRFBOCylAi9S/Ea7COQkQQN/rXhG8z1vCIUgLqG8/Ujk/Quwyuwj0BGYg6Ph7VrX8M3mOt5TShIkojLe0QhJtxf71IMj30EMpKggRMNu/ldU7wmFACeLXQE9hHISEQ/G2xpYwy7+V1TvCoUJEnEVYMSEOhv0bsUQ2IfgYwmsP94wEuGjep5VSgAdcEwcXgXvcswHPYRyHBEE4IGToQgetfHpHe9WgCSKODXI5PgY/au9G9v7COQ0QT0HA7J3/t2Wfa6UAAAf18Txg/mtRbaCvsIZDiCiODhk6Eqit6VdDivDAUAuHV8KnwsPFtoLfYRyIgCeo2AJTze64aOAC8NBUEQEOBnxnVc5dwq7COQIYkmhIy+HarqfWcJgJeGAgCIooCbr0yGvx83ymsp9hHIiGz9roQpMNwrtrRoine+6jN8LCZMHt1N7zI8EvsIZESCyYLQkbcAUPUuRTdeHQqSKGDSqCQE23z0LsWjsI9ARhU44GqIVpvXniUAXh4KACBJAqZcmaJ3GR6DfQQyKtHHipARN3l1IAAMBUiiiGuGJSIixE/vUjwC+whkVEFDfg3BbOzrLzeHSe8C3MUdE7rjtY8z9C7DrdX3EYpWv8M+wlmyS2vw/s58ZBVVo6TaBR+TiIQgH9zYKxxDOgVqt3s5PQdrDpc2un98oAULJl34bLW8xoVvs0qwOacCx8sckBUV8UE+uKFHOEZ1abjAqrDKibkbc7EnvwrhVjOm9Y9qUAcApGeXYd7mk3h7Ugr8OTUbkn8Qggb/2iunoJ6LoYC6rS+uHJiArzceQ2Z2id7luCX2Ec4vz+5EtVPBuKQQhPqZ4JAVpGeX49nvj+PhIbG4JiVUu61ZFPDYsLgG97eaL/5BtL+wCot35GNgXABuS4uAJAhIP16O2etP4HhZDe7qF6Xd9uWfclBU7cK0/lHYl1+Fv607gQWTkhEVULfnV62sYOG207i7XxQD4YyQK26BIPHjEGAoaGRZwcM398Mjr/wARfHemQdNYR/hwgbF2zAo3tbg2HWpYXjkq8NYtq+wQShIooCxXYMv+Tk6B/li4Vkf7ABwbWoonl59DP/bU4ibe0XA1yzC4VKw83Ql/j6hC/pE+WNiSij2FxzEtpN2rY7P9hbCapbwq+SQlr1gg7HEJMHWfzwEwQsuwNwMPFc6Q5JEJETbuFleE9hHuHSSKCDc34zK2sYLoGRFRWXtpe3PH22zNAgEoG4R5tBOgXAqKk7ZawEAtbIKFUDAmTMAQRDgb5HgcNXVUVjlxJI9hfjdoBiI/BAEICDi6t8CXridxfnwTOEcd1/TAz/tyEVJhUPvUtwC+wjNV+NU4JAVVDplbD5Rga25FRiZ2HC83+FScOPH++BwqQiwSBjdJQjT+kfBr4UbNJbUuAAAQT5197f5SIixWfDJ7gL85rIo7CuowpHiGswYVDeR4u1tpzEgLgB9onixKaBuoZpPDK+xcjaGwlkEQYBZEnHf9b3xj/e36V2O7thHuDQLtp3CyoN1PSlRAIYlBOKBQbHaz0P9TLipVzi6hflBUVVsO2nHisxiHCmuwZwJXSCJl/bNvcLhwjeHitE70opQ6y8r8x8ZEosX153AumNlAIBJPcLQK9If+/KrsPF4Od66PrkNXq3nE/1sCL3ybqiqyqGjszAUziFJIkZeFo9vNmdj16FCvcvRDfsIl25Sj3CMSAhCUbUT64+VQ1EB11n9qan9oxvcfnSXYMQFWrA4Ix/rs8swuktws59LUVXMWZ8De62CGYNiGvysX0wA3r0xBdmlDoRZTYjwt0BRVczfchKTe4YjKsCCFZlF+GJ/EVQAN/QIx8TU0KafyMDCxv0GosWXgXAO9hSaICsKHrypL0yS9/6ysI9w6ToF+eCy2ACMSwrBs1d2RrVTxqzvsqGq55+4cEOPcIgCsONU5SU915s/n8LWk3Y8NjQWXUMbr7HxM0voHmFFxJmrDK7OKkFxtQtTekcg46Qdb2/Lw9T+0bi3fzQWbDuFnae96+/Yr0sabGmjIXjZVdWag6HQBEkUERPmj8mjvfM0m/satY0RnYNwsKgaueW1572Nj0mEzUdCxSU0nj/YmY8VmcWY2j8KVyZdfAZRZa2MxRn5mNY/Gr5mET8cK8WIzoEYlhCIoQmBGJEQhO+PlDX7+T2dYLIgfOIDUJVLa/Z7C4bCeQiCgNsnpKJrnHddeYl9hLZTK9fNaKl0nv/Dp8opo7xG1hrFF/PlgSK8vzMfk3qEYUrviGbd58Nd+YgKMGPMmUVuxVUuhPn9MnIcZjWhqMrZrMcygpArpsBkC+NZwnkwFC7iqTsvh9nkHW8T+wgtU1rtanTMpahYe7gUPpKAhCAf1MoKqpoIh492FUAFcHmcrcF9T5Q5UHzOB/W6o2WYv+UUxnQJwvQB0WiOnHIHvjxQjN8NitHGzoP9TDhx1tnLiTIHQvy8o73oE98dQUOv58rlC/CO34QWkiQRseEBuOeanli4fI/e5bS7+j7CyY/YR7gUczflosqpoE+UP8KsJpRUu/D90TKcKHPg/gHR8DNLyLPX4qEVWRiVGIxOQXW78m47WYEtuXYMiA3A0E6/hEJRlRPTvziEcUnBeHJ4PAAgs7AK/0zPgc1HQr+YAHx/tOFwT48IK2JsDdcxAMB/tpzCyMQgpIZbtWMjOgfiue+PY9H20wCAzTkVmDXW+JenFX2siJr8JKCqgPe2Cy+KoXARoijg+lFJ2LI/DzsPFehdTrvheoSWG5kYhG+zSvBVZjHKHS74mSUkh/k22HPI3yJhULwNGafsWHOkBIoCxAZa8JvLonBjr/CLLiQ7XuqAS1FRViPj1Q25jX7+xLC4RqHwc04F9uRVYeGkhr2xwfGBuKdfFJYfqJt99JvLojAwruGKbCMKv2YGJP8gDhtdhKBeaGoEAaibjVRur8UDc76Dvdp4Y6+pCSGY89BwVB/Zjrwls/Uuh6jNBaSNQeR1D+ldhkfgwFozSKKIwAALZtyYpncpbY59BDI6U0gMwn91/wWnBtMvGArNJIl1i9pGXRZ38Rt7EK5HIEMTTYia/AQE0cRFas3EULgEiqLiwZv7ISbcGPvGcD0CGV3oqFthiUqEILGP0FwMhUsgigLMJhHP3DsYvh6+Dz3XI5DR+Sb2QdDQSV5/ec1LxXfrEpkkETHhAXj8tv56l9Ji7COQ0Um2UERNehxQuSX2pWIotIAkChiWFosbx3TTu5QWYR+BjEwwWRA95Y8QfQM4/bQFGAqtcM/EnrgstXlbDbgL9hHI6CKuexiWyM7sI7QQQ6EVVBX4v7sHIirUevEbuwH2EcjogkfcjICew7iNRSvwnWsFURTgY5bwl3sHw8fNG8/sI5DR+XcfgtBRt+pdhsdjKLSSSRIRH2nD47deBneeBs0+AhmZJaoLIq9/DCoby63GUGgD9Y3ne3/dW+9SmsQ+AhmZ5B+M6Fv/BIgip5+2Ab6DbUQQBFw/MsntZiSxj0BGJkhmRE2ZCckvkDON2ghDoY395tpeuHJggt5lADjTR5jOPgIZlCghcvIT8IlO4kyjNsRQaGOqquKRKf0wsEeU3qXgbw8Oh68P+whkQIKIiOsehjV5AGcatTG+m22sftOtmfcMRPfEi18/t73cPj4VqQmh7COQIYVf/VsE9BrBHkI74DvaDkRRgCSJmHX/UHSK6viLl6QmhOCWcewjkDGFXTUVgZeN466n7YSh0E4kUYCvWcILvxvWoYvb6voIg9lHIEMKGXUrggZdq3cZhsZQaEeSJCLI34J/PHwFYjtou+26PoKJfQQynKChNyBkxM16l2F4DIV2JkkiAv0tmPPwFe0+lMQ+AhlV4ICrETb2Tr3L8AoMhQ4gSSIC/Mz4+0MjkBgT2C7PwT4CGZWt/3iET7hP7zK8BkOhg0iSCKuPCbMfHIGkuKA2fWz2EciogkfchIirf6t3GV6FodCBJEmEr4+Elx4cgZSEtpuuyj4CGY+AsPH3InTUbXoX4nUYCh1MEkVYzCJenDEMPbuEtvrx2EcgwxFNiLzhcQQOuFrvSrwSQ0EHkijCbBLx/G+HYVifmBY/DvsIZDSC2RfRt/4R/t2Hch2CTgRVVVW9i/BWiqJCFAW8s2Ivln6fdUn3tZhEvD9rPMxOO3IWPMFhI/J4op8NMbc9A0tUZ25upyOeKehIFOu+CU29thcevrkvJLH534zYRyAjkQLDEfeblxgIboCh4CbGDeqMZ6cPhb+v6aK3ZR+BjMQnNhlx0/4OU3AkA8ENMBTchCgK6N01DC8/NuqC22Kwj0BGYut3JWLvfgGSn42B4CbYU3AzsqygyuHCsws3ITO7pMHP2Ecgw5BMCB9/LwL7j4eqqmwquxGGghuSFQWqAry5dBe+3ZytHf/no1cgJT4IJxf/icNG5LEkWyiibvpD3cVxeC0Et8NQcFP1356+23oCb3y6EzeO6YbbJnRH0ep3OGxEHssnvjuib/4/iL7+HC5yUwwFNycrKvKKKhEd6ofqI9uRt2S23iURtUjg5b9C2PhpAMBAcGMXn+pCupJEAVFhVoiiiOojO/Uuh+iSCRZfhE+4H7a00XqXQs3AMwUPoaoKBEFExa4fULhqAVRnjd4lEV2Ub6eeiJz0KKSAEJ4deAiGgodRFRmuskLkLf0Hak8f1bscoiYJkhkho25D0JBfA6rCQPAgDAUPpCoyAKA0fSlK0j8FZJfOFRH9whLVBZGTHoc5NIazizwQQ8GDqaoCZ/FpFHz5Ohy5mXqXQ95OEBE87AaEjLzlzB95duCJGAoeTlVkQBBRvnUlir//kL0G0oU5NAaR1z8GS0wSF6J5OIaCQaiKAtlegoKv3kD1kR16l0PeQjQhaNDEuovhCCIEiWcHno6hYCCqokAQ62YoFa15B0o1t8Gg9uPXtR/Cf3U/TMFRPDswEIaCAamKDKWmCkVr3oF9948A+FdMbccUHIWw8VPhnzwQqiKzd2AwDAWDqj9rcOQdQ9Hqd1CTvUfvksjDCSYLgodNRvCwGwAIHCoyKIaCwdV/k6s6nIGiNYvhLDyhd0nkgfy7D0HY+Hsh+QdzmqnBMRS8hCrLgCigYsdalPz4MWR7qd4lkQewRHdF2Lh74Ne5t3b2ScbGUPAyqiJDlV0o3bgMZZuWQ3U69C6J3JA5IgGho2+Df8ogqLLMoSIvwlDwUqqiQKmuQMlP/0PFjrVQXbV6l0RuwBwWh5CRt8C/xzBAURgGXoih4MVUVQEgQKmpRNnPK1C+7WtOY/VSlsjOCB5+45kwkCFI3EDZWzEUCEDdmYOquFCx/VuUbv4Scnmh3iVRB7DEJCFkxM3wTxkIVXYxDIihQA3Vb7Zn3/sTSjcug7OAs5UMR5TgnzoYgQOuhl9CT/YMqAGGAjWp/oOiKmsbSjct5zoHA5BsoQi87CoE9p8AyT+IC8+oSQwFuqD6cHCW5qNixxpU7PoeckWx3mXRJfDt3BuBl/8K/qmDAagMAroghgI1i6qqgKoAgoDqIztRsWMNKg9uBRRey8EdCRY/2NJGI3DANbCExXKIiJqNoUCXrH7YQa62w77re5Tv/A7OguN6l+X1BLMPrN0uh3/P4fDvdjlwpmnMzeroUjAUqFXqv4E6Th+Bfe9PqDq4Bc7ik3qX5TUEkwXWbv3rgiB5IASTuUPOChYtWoRFixYhLy8PY8aMwRtvvNGuz9cc8+bNw3//+19kZGToXYpH4/wzapX6Dx9LVCJCIxMRduXdcJacRuWBTag8uAWO3IN1w07UZgSTBX5J/RDQYzisKQMhmn0aBEF7B8KxY8cwe/Zs3H///RgzZgxCQkLa9fmoYzEUqE0IggicGaUwh0QjaNB1CB46CXK1HVUHf0blwS2oPrqT22q0kDksDn6JfeDXJQ1+Xft1eBCc7ejRo1BVFVOmTEGnTp067HmpYzAUqF3Uf0hJfgEI6D0Str5jocouVB/bjersvag5sR+1pw5DlZ06V+qeJFso/BLT6oIgqR9M/sF1K9DVX2YP6dE4njlzJpYtWwYAGDduHADgpZdewrhx4/DKK69gzZo1KC0tRUpKCp544gmMGDFCu+9dd90Fq9WK6667DnPnzkVeXh6GDh2Kv//977Db7XjmmWewfft2xMbG4plnnsHgwYO1+37++ef45JNPcPjwYaiqiu7du+Opp55CWlraBestLy+/aF3UEHsK1KFURUH9tEhVdsFx+ghqsveiJucAanIOeO02G5ItFL6xKfDt0gfWrpfBHBIFAG63yvj48eP45ptv8M9//hOvv/46IiIiEB8fj9/+9rcoKirCQw89hKioKCxfvhwrV67E0qVLkZqaCqAuFI4dO4aYmBhMnz4ddrsdL7zwAoYNG4bc3FxMmjQJXbp0wVtvvYXMzEx8//338Pf3BwDtuRISElBbW4uvvvoKX3/9NZYvX44uXboAaNxTqK2txW233XbRuqgh9/ltI69w9tbLgmSCT2wyfKK7nrlwC+AsPoXq7D1w5B5EbcEJ1BblQnVU6VVu2xNEmENjYInqAp/oLvCJ7gpLdFdIfgEAGoeAOwUCACQkJGgfwj169EB8fDw+++wzHDhwAF988QW6desGALjiiiuQnZ2NN954A//617+0+9vtdsyfPx+hoaEAgMzMTPz3v//FrFmzcNtttwEAIiMjcd1112Hjxo3a2chDDz2kPYaiKBg+fDh27dqFZcuW4Yknnmiy1i+//LLZddEv3Os3jryOIAja1EkAMIfGwBQUAVu/cdpUSrmyDLUFx+tCojAHzsIcOItyIFeW6VX2xUkmmALDYQoMhzk0pi4AYpJgjugM0WQGUBcAEKUGU0bdLQSaIz09HSkpKUhMTITL9cu6lWHDhmH58uUNbtu9e3ctEAAgMTFRu+25x06fPq0dO3z4MF555RVkZGSgqKhIO37s2LE2qYt+4Xm/gWR4534wSv5B8LX2hm+nHg0+RBVHFWqLTsJVlg+5ogQuezHkihLI9hLIVWWQq8ohV1cAchsusBNEiBZfCBY/SNZAmIIi6j78g8JhCoqAOTgKpqAISNZA7S6qqja586gnBkBTSkpKsG/fPvTq1avRz6Rz+h6BgYEN/mw21wWkzWbTjlksFgCAw1E3KcFut2PatGkIDQ3FzJkzERsbCx8fH/z5z3/WbtPauugXxvitJMM794wCAEQfK3xju0GN7gqoMgCxyearUlsDpaYSqqsWquyCqrigynUXG4LsrDsmO+uuTqfIgChB9LGe+cfvTAj4QjT5QDjzLf9sqqKcWe0tNnllsqZqN5KgoCCkpqbixRdfbJfH37FjB06fPo233noL3bt3145XVFQgOjpat7qMyri/qeQ16j6Iz3+ZSNHiC9Hie8HHqJtvoQKqCghC3RTbNnp+oxs2bBjWrVuHyMhIREVFtfnj19TUAPjlrAIAtm/fjtzcXCQnJ+tWl1ExFIhQvxWEoK21oOabNGkSPv74Y9x9992YNm0aEhMTUVFRgX379sHpdOLJJ59s1eP369cPVqsVzz77LKZPn468vDzMmzfvoh/07V2XUTEUiKhVLBYL3n33XcybNw/z589HQUEBgoOD0bNnT9x+++2tfvzw8HD861//wpw5c/DAAw8gMTERzz77LBYuXKhrXUbFdQpERKTx3oFQIiJqhKFAREQahgIREWkYCkREpGEoEBGRhqFAREQahgIREWkYCkREpGEoEBGRhqFAREQahgIREWkYCkREpGEoEBGRhqFAREQahgIREWkYCkREpGEoEBGRhqFAREQahgIREWkYCkREpGEoEBGRhqFAREQahgIREWkYCkREpGEoEBGRhqFAREQahgIREWkYCkREpGEoEBGRhqFAREQahgIREWkYCkREpGEoEBGRhqFAREQahgIREWkYCkREpGEoEBGRhqFAREQahgIREWkYCkREpGEoEBGRhqFAREQahgIREWn+Hys9wILW2RZEAAAAAElFTkSuQmCC"/>

<img src="data:image/png;base64,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"/>

위와 같이 남성이 여성보다 배에 많이 탔으며, 남성보다 여성의 생존 비율이 높다는 것을 알 수가 있다.

이제 사회경제적 지위인 `Pclass`에 대해서도 그려보자.



```python
pie_chart('Pclass')
```

<img src="data:image/png;base64,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"/>

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAa0AAACXCAYAAACmwARIAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABQi0lEQVR4nO2dd3xUVfbAv2/6pEx6DwkhJKH33kERBFFh9cfaXbHXdXUVddVdLGBZO3YFxY6yKkWlSxMBkaK0EEp675lMe+/9/hgyMKRDSPLC+34++cC8d++7Z2bu3HPPueeeK8iyLKOioqKioqIANG0tgIqKioqKSlNRlZaKioqKimJQlZaKioqKimJQlZaKioqKimJQlZaKioqKimJQlZaKioqKimJQlZaKioqKimJQlZaKioqKimJQlZaKioqKimJQlZYCmD17NhMmTGhrMVTaELUPqIDaD6ADKq3U1FTuvfdeLrjgAvr27cvQoUO55pprWLt2ba2y1113HbNnz24DKWuTl5fH66+/zv79+9taFMVTVVXFa6+9xqxZsxgyZAgpKSksWbKkzrJqH+i47Nmzhzlz5jB16lT69evHuHHjuO+++zh69Gitsmo/UA66thagpcnOzqaqqorp06cTHh5OdXU1K1eu5I477mDOnDnMnDmzrUWsk/z8fN544w1iYmLo3r27172nnnoKNUVk0ykpKWH+/PlER0eTkpLCtm3b2lqkJqH2gZbl/fffZ+fOnUyePJmUlBQKCgr49NNPmTFjBl9++SXJycltLWKdqP2gYTqc0ho7dixjx471unbttdcyY8YMFixY0GpKy263o9fr0WjO3pjV6/UtINH5Q3h4OJs2bSIsLIy9e/dyxRVXtIkcah9oW2688UZefPFFDAaD59qUKVOYNm0a7777Li+++GKryKH2g5alw7kH60Kr1RIVFUVFRUWjZRctWsTUqVPp27cvgwcPZsaMGSxdurTBOr/++ispKSksX76cl19+mdGjR9O3b18qKyspLS3lueeeY9q0afTv358BAwZw8803c+DAAa/6NQPrI488QkpKipdLqy4/ttVqZd68eYwdO5ZevXoxadIkPvjgg/N+FgZgMBgICws74/pqH+gYDBgwwEthAXTu3JmkpCSOHDnSaH21H7RPOpylVYPVasVms1FZWcnatWvZsGEDF198cYN1vvrqK55+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"/>

위와 같이 Pclass가 3인 사람들의 수가 가장 많았으면 Pclass가 높을 수록 생존 비율이 높다는 것을 알 수 있다.

마지막으로 어느 곳에서 배를 탔는지를 나타내는 `Embarked`에 대해서 살펴보자.



```python
pie_chart('Embarked')
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYUAAAGbCAYAAAAr/4yjAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABSLElEQVR4nO3dd3hUdfr38fc5U9J7JyEhARJKCEXpvYkIoujaUbG3VVdX17buWvax/bCBurqWtSGWBTuIgoiVIhZ6hxRI733KOc8fISOhl0nOlPt1XVzKlHPuGZL5zLceRdd1HSGEEAJQjS5ACCGE55BQEEII4SKhIIQQwkVCQQghhIuEghBCCBcJBSGEEC4SCkIIIVwkFIQQQrhIKAghhHCRUPAD48aN47rrruuQcy1YsICsrCzWrVvXrufJyspizpw57XqO9rBy5UqysrJYuXKlRx7PU9x9992MGzfO6DL8koSCQVo/PA/357fffjO6RK/S+n56irlz57JgwQKjyzghL774IkuWLDnh52/fvp05c+ZQUFDgxqpERzEbXYC/u+WWW0hJSTno9tTUVAOqEe4yb948oqKiOOecc9rcPnDgQNauXYvFYnHLedx9PICXXnqJSZMmMWHChBN6/vbt23nuuecYNGjQIX+2hWeTUDDYqFGj6NOnj9FlnLTm5ma3fjB5El3XaW5uJjAw8KSPpaoqAQEBbqiqfY4nhHQfebiCggKysrJ49dVXmTt3LuPHj6dv375ceeWVFBYWous6zz//PKNGjSInJ4cbbriBqqqqQx7r+++/56yzzqJPnz6cccYZfPnll23ur6qq4vHHH+fMM8+kf//+DBgwgKuvvprNmze3eVxrP/bnn3/O008/zciRI+nbty91dXWHPG91dTV/+tOfGDVqFDt37gTAZrMxe/ZsJk6cSHZ2NqNHj+aJJ57AZrO1ea7NZuORRx5hyJAh9O/fn+uvv56ioqJjeu/WrVvHVVddxeDBg8nJyWHcuHHcc889R31e6xjMd999xznnnENOTg7vvvsuAPPnz+eyyy5j6NChZGdnc8YZZ/DOO+8c9Pxt27axatUqV3fgpZde2ua9O3AMYNGiRa5zDR48mDvuuIPi4uKj1nqo41166aVMnTqV7du3c+mll9K3b19GjhzJyy+/fNTjZWVl0dDQwIcffuiq/e6773bdv3HjRq6++moGDBhA//79ufzyy9t0dS5YsIBbb70VgMsuu8x1jNb6lixZwrXXXsuIESPIzs5mwoQJPP/88zidzqPWJjqGtBQMVldXR0VFRZvbFEUhKiqqzW2ffvopdrudSy+9lKqqKl555RX+8pe/MGTIEFauXMk111xDbm4ub7/9No8//jiPPvpom+fv3r2b2267jQsvvJDp06czf/58br31Vl555RWGDx8OQH5+PkuWLOH0008nJSWFsrIy3nvvPWbMmMHnn39OQkJCm2O+8MILWCwWrrrqKmw22yFbChUVFVx55ZVUV1fz9ttvk5qaiqZp3HDDDaxZs4bzzz+frl27snXrVt544w12797NCy+84Hr+fffdxyeffMLUqVMZMGAAK1as4Nprrz3q+1peXs5VV11FVFQU1157LeHh4RQUFPDVV18d9bkAu3bt4q9//SsXXHAB559/Punp6UBLt1D37t0ZN24cZrOZZcuW8eCDD6LrOpdccgkA9957Lw8//DDBwcFcf/31AMTGxh72XAsWLOCee+6hT58+3H777ZSXl/Pmm2/yyy+/8NFHHxEeHn5MNe+vurqaq6++mokTJzJ58mQWL17MrFmzyMzMZPTo0Yd93hNPPMHf//53cnJyOP/884E/ujK3bdvGJZdcQkhICFdffTVms5n33nuPSy+9lLfffpu+ffsycOBALr30Ut566y2uv/56MjIyAOjatSsAH374IcHBwVxxxRUEBwezYsUKZs+eTV1dHXfddddxv07RDnRhiPnz5+uZmZmH/JOdne16XH5+vp6ZmakPGTJEr6mpcd3+5JNP6pmZmfq0adN0u93uuv3222/Xe/furTc3N7tuGzt2rJ6ZmakvXrzYdVttba0+fPhw/eyzz3bd1tzcrDudzjZ15ufn69nZ2fpzzz3num3FihV6ZmamPn78eL2xsfGQr2vt2rV6SUmJPmXKFH38+PF6QUGB6zEfffSR3qNHD3316tVtnjtv3jw9MzNTX7Nmja7rur5p0yY9MzNTf+CBB9o87vbbb9czMzP12bNnH/b9/eqrr1x1HK/W9+vbb7896L4DX6+u6/qVV16pjx8/vs1tU6ZM0WfMmHHQY1vfuxUrVui6rus2m00fOnSoPnXqVL2pqcn1uGXLlumZmZn6s88+e8RaDzyeruv6jBkz9MzMTP3DDz903dbc3KwPHz5cv/nmm494PF3X9X79+ul33XXXQbffeOONeu/evfW8vDzXbcXFxXr//v31Sy65xHXbokWLDqqp1aHev/vvv1/v27dvm5/Zu+66Sx87duxRaxXuJ91HBvvHP/7Bf//73zZ/DtXMP/300wkLC3P9PScnB4Bp06ZhNpvb3G632w/qeoiPj2fixImuv4eGhnL22WezceNGSktLAbBarahqy4+E0+mksrKS4OBg0tPT2bhx40E1nX322YftZy8uLmbGjBnY7Xbmzp1LcnKy674vvviCrl27kpGRQUVFhevPkCFDAFxdDcuXLwdwdb20uvzyyw95zv21vlfffPMNdrv9qI8/UEpKCiNHjjzo9v1fb21tLRUVFQwaNIj8/Hxqa2uP+zzr16+nvLyciy66qM3YwJgxY8jIyOCbb7457mMCBAcHc9ZZZ7n+brVa6dOnD/n5+Sd0PKfTyQ8//MCECRPo3Lmz6/b4+HimTp3KmjVrDtt9uL/937/WVvKpp55KY2Ojq2tRGEu6jwyWk5NzTAPNSUlJbf7e+qF3uNurq6vb/PKmpaWhKEqbx3bp0gWAPXv2EBcXh6ZpvPnmm7zzzjsUFBS06eeNjIw8qKYjzSy58847MZvNLFy4kLi4uDb35ebmsmPHDoYOHXrI55aXl7vqUlX1oJlYrV0SRzJo0CAmTZrEc889x+uvv86gQYOYMGECZ555Jlar9ajPP9xrW7NmDXPmzOG3336jsbGxzX21tbVtgvtY7N27F8DVPbW/jIwM1qxZc1zHa5WYmHjQv3dERARbtmw5oeNVVFTQ2Nh4yDq7du2KpmkUFhbSvXv3Ix5n27ZtPPPMM6xYseKgEDmRUBXuJ6HgJUwm0yFvb/1mfyD9BK6y+uKLL/Lss89y7rnncuuttxIREYGqqjzyyCOHPN6RZuOcdtppfPTRR7z55pv89a9/bXOfpmlkZmYedtA3MTHxuGs/kKIozJ49m99++41ly5bx3Xffce+99/Lf//6X9957j5CQkCM+/1CvLS8vj5kzZ5KRkcHdd99NUlISFouF5cuX8/rrr6Np2knX7S6H+3kxUk1NDTNmzCA0NJRbbrmF1NRUAgIC2LBhA7NmzfKo98+fSSj4idzcXHRdb/Ptcffu3QCurp3FixczePBgHnnkkTbPrampOWjg+2hmzJhBamoqs2fPJiwsrM3gcGpqKps3b2bo0KEHfZvdX3JyMpqmkZeX16Z1cDzdDP369aNfv37cdtttfPrpp9xxxx0sXLiQ884777heD8DXX3+NzWbj3//+N506dXLdfqjVxEd6XftrPc6uXbsOajnt2rWrzXmMFB0dTVBQELt27Trovp07d6KqqqvVerjXvmrVKqqqqnjuuecYOHCg63ZZ5OZZZEzBT5SUlLSZeVNXV8dHH31Ez549Xd07JpPpoBbBokWLjmlq5KHcdNNNXHnllTz55JNtpm1OnjyZ4uJi3n///YOe09TURENDA9CyhgPgrbfeavOYN95446jnrq6uPui19OzZE+Cgaa/HqvXb9/7Hra2tZf78+Qc9NigoiJqamqMeMzs7m5iYGN599902dS1fvpwdO3YwZsyYE6r1ZAQHBx9Uu8lkYvjw4SxdurTNh3hZWRmfffYZp5xyCqGhoUDLa4eDu4NaW7X7v382m+2gKb3CWNJSMNi33357yG++AwYMaDMmcLK6dOnCfffdx7p164iJiWH+/PmUl5e3mbo6ZswYnn/+ee655x769+/P1q1b+fTTT0+qjrvuuou6ujoeeughQkJCOOusszjrrLNYtGgR//znP1m5ciUDBgzA6XSyc+dOvvjiC1555RX69OlDz549mTp1Ku+88w61tbX079+fFStWkJube9Tzfvjhh8ybN48JEyaQmppKfX0977//PqGhoa6wOV7Dhw/HYrFw/fXXc+GFF1JfX88HH3xATEyMa7C+Ve/evZk3bx4vvPACaWlpREdHH3IMxWKxcMcdd3DPPfcwY8YMpkyZ4pqSmpyczMyZM0+o1pPRu3dvfvrpJ/773/8SHx9PSkoKffv25S9/+Qs//vgjF198MRdffDEmk4n33nsPm83GnXfe6Xp+z549MZlMvPzyy9TW1mK1Wl3rTCIiIrj77ru59NJLURSFjz/++IS6OkX7kVAw2OzZsw95+6OPPur2ULj//vt54okn2LVrFykpKa6FZ62uv/56Ghsb+fTTT1m4cCG9evXipZde4sknnzypcz/44IM0NDRw7733EhIS4lqw9Prrr/Pxxx/z1VdfERQUREpKCpdeemmbwcxHHnmEqKgoPv30U5YuXcrgwYP5z3/+c8S59tAy0Lxu3ToWLlxIWVkZYWFh5OTkMGvWrBN+XzMyMpg9ezbPPPMMjz/+OLGxsVx00UVER0dz7733tnnsTTfdxN69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"/>

<img src="data:image/png;base64,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rVO9Nm3axIkTJxg0aFC96tqUcAup8PXXX6e0tJRhw4bRqlUrLBYLu3fvZuXKlURERFRaiV6R0aNHEx8fT8eOHfH19WX//v38/vvvjB8/vt71GTRoEL/88gteXl60adOGhIQE/vrrL3x9fZ3StW/fHpVKxeeff05hYSFarZZevXpVufZs3LhxfPfdd0ybNo2DBw8SERHB77//zu7du5k+fbpjnudapXnz5syaNYunn36aUaNGOXbOMJvN7Nmzh1WrVintoAnTokUL3nnnHZ599lluuukmxo4d67RzRkFBAe+///4lFx//9NNPLF68mKFDh9K8eXOKi4v5/vvv8fLyYsCAAfWqV9++fdFoNDz66KPcddddFBcX88MPPxAQEEBmZqZT2o4dO7J48WI+/vhjWrRogb+/P717966Up0aj4bnnnuPFF19k/PjxjB492hEOHxERwaRJk+pV16aEWxiuqVOnsmrVKjZt2sR3332HxWIhPDyce+65h8cee+ySaxomTJjA+vXr+fPPPzGbzYSHh/N///d/TJ48ud71eemllxBFkV9//ZWysjK6du3KvHnzePDBB53SBQUF8eqrr/LZZ5/x0ksvYbPZ+Oqrr6rssHQ6HQsXLmTWrFn89NNPFBUVERUVxVtvvXXJDvla4vrrr2fZsmV88cUXrFu3jsWLF6PVaomJiWHatGnceeed1T6rtAP3Z8SIEURFRfHZZ5/x448/kp2djSRJeHh4sHTp0moDtMqJj49n//79/Pbbb2RlZWE0GomLi2PWrFn13m2jVatWfPjhh/znP//hnXfeITAwkLvvvht/f3+mT5/ulPaJJ57g3LlzzJ07l+LiYuLj46s0XABjxoxBp9Px+eefM2vWLAwGA0OHDuX5559X1nABgqzMIisoKLgpP//8M9OmTePmm2/m3XffdXV1FK4SbjHiUlBQUKiKW2+9lYyMDN577z1CQ0N55plnXF0lhauAMuJSUFBQUHAr3CKqUEFBQUFBoRzFcCkoKCgouBWK4VJQUFBQcCsUw6WgoKCg4FYohktBQUFBwa1QDNc1yrJlyxg7dizdunWja9eu3HDDDbz00ktkZ2df1XoMGTKEmTNn1umZadOmOR37cTmsXbuWmJgYp1OLFRQUGjfKOq5rkM8//5z33nuPSZMmMWXKFGRZJjExkV9//ZWMjIwqd3S4UsyZM6fO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"/>

위와 같이 Southampton에서 선착한 사람이 가장 많았으며, Cherbourg에서 탄 사람 중에 생존한 사람의 비율이 높았고, 나머지 두 선착장에서 탄 사람들은 생존한 사람보다 그렇지 못한 사람이 조금 더 많았다.


## 3.2. 범주형 특성에 대한 Bar chart


이번에는 아래의 특성들에 대허서 Bar chart를 정의해서 데이터를 시각화 해보자.



```python
def bar_chart(feature):
     survived = train[train['Survived']==1][feature].value_counts()
     dead = train[train['Survived']==0][feature].value_counts()
     df = pd.DataFrame([survived,dead])
     df.index = ['Survived','Dead']
     df.plot(kind='bar',stacked=True, figsize=(10,5))
```

먼저`SibSp`에 대해서 Bar chart를 그려보자.



```python
bar_chart('SibSp')
```

<img src="data:image/png;base64,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/ADOqwFakPz8fG3dulVbt27VZ599pqioKF1zzTVWxwIABMBrr72me+65R9dee628Xm+tdT179tThw4ctSgY0f5QiIMidOnVKb775prZu3ap9+/apVatWGjVqlO6++26lpKRwTREA2MTp06eVkJBwznVer1eVlZUBTgQED/5vCQhyw4YNk9Pp1E9/+lM98cQTGjFihMLDw62OBQAIsF69emnPnj0aOnRonXXvv/++evfubUEqIDhQioAg9/DDD+sXv/iF2rZta3UUAICFbrrpJs2bN08hISEaPXq0pKrTqvft26cXX3xRmZmZFicEmi9mnwMAAGghnnvuOS1fvlylpaU1Ey1ERERoxowZuvnmmy1OBzRflCIgCD388MO65ZZb1LVrVz388MM/uv3cuXMDkAoA0ByUlJRo7969KiwsVFRUlJKSkhQZGWl1LKBZ4/Q5IAjt3LlTkyZNUteuXX90ilWHw0EpAoAW7siRI9q0aZP27dunEydOyOFwqEOHDrr44ovVu3dvShHwIzhSBAAAEMT++te/6ne/+53Ky8vVqVMndenSRT6fT/n5+Tp27JjCw8OVmZmptLQ0q6MCzRalCAhyhw4dUq9evayOAQCwwMGDB3XVVVfpkksu0bx58+pMyf3VV1/poYce0r59+/SXv/xFcXFxFiUFmjfD6gAAGmf06NG6+uqr9fzzz+vYsWNWxwEABNAf//hH9ejRQ6tXrz7nPYp69+6ttWvXqnv37tq4caMFCYHgQCkCgtyqVasUFxenZcuWacSIEbrhhhv06quv6vTp01ZHAwD42fvvv69rrrlGYWFh37tNWFiYrrnmGr3//vsBTAYEF06fA1qIsrIy7dy5U1lZWcrOzpZpmkpNTdW4ceM0btw4q+MBAPzg0ksv1fLly895w9bvysnJUXp6uv7xj38EKBkQXJh9DmghWrVqpbS0NKWlpam4uFhvvfWWnnzySb3zzjuUIgBooUpKStSmTZsf3a5169Y6c+ZMABIBwYlSBLQwH330kbKysvTmm2+qoKCAi2oBoAXjhB+gaVCKgBbgwIEDeuONN/Tmm2/q66+/VpcuXTR27FiNGzdO/fr1szoeAMCPbrzxRjkcjh/chvIE/DBKERDkxo8frwMHDig6OlqjR4/WwoULdckll1gdCwAQAOnp6VZHAFoEJloAgtz999+vsWPHaujQoXI6nVbHAQAACDpMyQ0EsbNnz6qwsFDh4eEUIgAAgAaiFAFBLDw8XB988IG8Xq/VUQAAAIIWpQgIcikpKdqzZ4/VMQAAAIIWEy0AQe7qq6/W/PnzVVJSop/+9KeKiYmpMwvRgAEDLEoHAADQ/DHRAhDk+vbtW+vxdwuRz+eTw+HQZ599FuhYAAAAQYMjRUCQW79+vdURAAAAghpHigAAAADYGhMtAAAAALA1Tp8Dglzfvn3rTKzw/8c1RQAAAN+PUgQEufvuu69OKfJ4PNqzZ48KCgo0ZcoUi5IBAAAEB64pAlqw//mf/1G3bt109913Wx0FAACg2eKaIqAFmzBhgl555RWrYwAAADRrlCKgBcvLy5NpmlbHAAAAaNa4pggIcs8991ydZRUVFTp48KC2bdumcePGWZAKAAAgeHBNERDk+vbtW2dZWFiYOnfurCuvvFJ33XWXIiIiLEgGAAAQHChFAAAAAGyNa4oAAAAA2BrXFAFB6NSpUyooKKhz6tznn3+up556SgcPHlSHDh104403auTIkRalBAAACA6cPgcEoblz5+qTTz7Ra6+9VrPs6NGjmjBhgsrKypSYmKj8/HydPn1aL7zwgv7rv/7LwrQAAADNG6fPAUHon//8p8aPH19r2fPPP68zZ87omWee0Z/+9Cft3LlTF110kdasWWNRSgAAgOBAKQKC0LFjx9S7d+9ay3bt2qV+/fopNTVVktSqVStdf/31+uKLL6yICAAAEDQoRUAQcjgccjgcNY9PnDihI0eO1DlNrlOnTiosLAx0PAAAgKBCKQKCUFxcnN59992ax7t27ZLD4VBKSkqt7Y4fP6727dsHOh4AAEBQYfY5IAjdcMMNmj17tjwejzp06KCXXnpJPXv21OWXX15ru927d6tPnz4WpQQAAAgOlCIgCE2YMEHHjh3Thg0b5PF4NGDAAD3wwAMKCfnPV/rkyZPatWuXpk+fbmFSAACA5o8puQEAAADYGtcUAQAAALA1ShEAAAAAW6MUAQAAALA1ShEAAAAAW6MUAQAAALA1ShEAAAAAW6MUAQAAALA1ShEAAAAAW/v/AGfxSRGt2bIoAAAAAElFTkSuQmCC"/>

위와 같이 2명 이상의 형제나 배우자와 함께 탔을 경우 생존한 사람의 비율이 컸다는 것을 볼 수 있고, 그렇지 않을 경우에는 생존한 사람의 비율이 적었다는 것을 볼 수 있다.



```python
bar_chart('Parch')
```

<img src="data:image/png;base64,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wYUV+vfvb0Ei4OJHKQKC3LvvvqtHH31UCQkJ8nq9tbZ16dJFx48ftygZACCQKioqtGDBAv35z3+usz+o9tlnnwU4FRAcuE8REOS8Xq9at259wW0ej0ehoaEBTgQAsMLq1au1Z88ePf744/L5fJo7d64WL16soUOHqlu3bvrNb35jdUTgokUpAoLc5ZdfrldfffWC27Zt26Yrr7wywIkAAFbYvn27pk6dqlGjRkk6v3+46aab9Nxzz+mqq67Szp07LU4IXLwoRUCQy8jI0K5du3TbbbfpxRdflMPh0Ntvv63p06dr586dmjZtmtURAQABUFBQoLi4ODmdToWHh8vj8dRsGzt2rLZv325hOuDiRikCglxycrI2bdokh8OhJ554Qj6fT7/5zW904sQJbdy4kYtqAcAmYmNja4pQ9+7d9f7779dsO3z4sEWpgODAQgtAC5CcnKzNmzfr7NmzcrvdcrlcioiIsDoWACCABg0apI8++kjDhw/XhAkTtGTJEuXm5io0NFRvv/22xowZY3VE4KLFfYqAIPfKK6/ohhtukMvlsjoKAMBCJ06cUFFRkfr06SNJ2rhxo7Zv365z587pmmuu0c9//vNvXZgHsDtKERDkLrvsMjkcDqWkpOjGG2/Uddddp1atWlkdCwAAIGhQioAg53a79eabb2rbtm368MMPFR4eruHDh2vMmDEaNmyYQkI4SxYA7MTtduurr77S119/rbS0NEVFRencuXMKDQ2VYXA5OXAhlCKgBTlx4oSysrL0xhtvaN++fYqKitLIkSP161//2upoAAA/M01Ty5cv1wsvvKDy8nI5HA5t3bpV/fv31+TJk3XFFVdo6tSpVscELkr85wKgBYmNjdUdd9yhl156SevWrVN4eLi2bNlidSwAQAA888wz2rx5s2bOnKk333xT3/zv3sOHD+c+RcB34LwaoAUpKCjQtm3btG3bNn322WeKiorSLbfcYnUsAEAAvPbaa3r44Yd16623yuv11trWs2dP5efnW5QMuPhRioAgd/r0ab3xxhvatm2b9u3bp1atWmnEiBF68MEHlZKSwjVFAGATZ86cUUJCwgW3eb1eVVVVBTgREDz41xIQ5IYNGyan06kf/OAHevrpp3XttdcqPDzc6lgAgADr1auX9uzZo6FDh9bZ9sEHH6h3794WpAKCA6UICHILFy7Uj370I7Vt29bqKAAAC915552aO3euQkJCdMMNN0g6f1r1vn379MILL2jx4sUWJwQuXqw+BwAA0EJs2LBBK1euVHl5ec1CCxEREZo+fbomTZpkcTrg4kUpAoLQwoULddddd6lr165auHDh946fM2dOAFIBAC4GpaWl2rt3r4qKihQVFaXk5GRFRkZaHQu4qHH6HBCEdu7cqfHjx6tr167fu8Sqw+GgFAFAC3f06FFt2bJF+/bt08mTJ+VwONShQwddeeWV6t27N6UI+B4cKQIAAAhif/nLX/SrX/1KFRUV6tSpk7p06SKfz6eCggIdP35c4eHhWrx4sdLT062OCly0KEVAkDt8+LB69epldQwAgAUOHTqkm266SVdddZXmzp1bZ0nur776So8++qj27dunP/3pT4qLi7MoKXBxM6wOAKBpbrjhBt18883auHGjjh8/bnUcAEAA/e53v1OPHj20du3aC96jqHfv3lq3bp26d++uF1980YKEQHCgFAFBbs2aNYqLi9OKFSt07bXX6vbbb9crr7yiM2fOWB0NAOBnH3zwgW655RaFhYV965iwsDDdcsst+uCDDwKYDAgunD4HtBBnz57Vzp07lZWVpezsbJmmqdTUVI0ZM0ZjxoyxOh4AwA+uvvpqrVy58oI3bP2mnJwcTZ06VX/7298ClAwILqw+B7QQrVq1Unp6utLT01VSUqI333xTzzzzjN59911KEQC0UKWlpWrTps33jmvdurXKysoCkAgITpQioIX5+OOPlZWVpTfeeEOFhYVcVAsALRgn/ADNg1IEtAAHDx7U66+/rjfeeENHjhxRly5dNHr0aI0ZM0ZJSUlWxwMA+NEdd9whh8PxnWMoT8B3oxQBQe7GG2/UwYMHFR0drRtuuEGLFi3SVVddZXUsAEAATJ061eoIQIvAQgtAkPvlL3+p0aNHa+jQoXI6nVbHAQAACDosyQ0EsXPnzqmoqEjh4eEUIgAAgEaiFAFBLDw8XB9++KG8Xq/VUQAAAIIWpQgIcikpKdqzZ4/VMQAAAIIWCy0AQe7mm2/WvHnzVFpaqh/84AeKiYmpswpR//79LUoHAABw8WOhBSDI9e3bt9bjbxYin88nh8Ohzz77LNCxAAAAggZHioAgt2nTJqsjAAAABDWOFAEAAACwNRZaAAAAAGBrnD4HBLm+ffvWWVjh/8c1RQAAAN+OUgQEuVmzZtUpRR6PR3v27FFhYaEmTpxoUTIAAIDgwDVFQAv2P//zP+rWrZsefPBBq6MAAABctLimCGjBxo4dq5dfftnqGAAAABc1ShHQguXl5ck0TatjAAAAXNS4pggIchs2bKjzXGVlpQ4dOqTt27drzJgxFqQCAAAIHlxTBAS5vn371nkuLCxMnTt31siRIzVlyhRFRERYkAwAACA4UIoAAAAA2BrXFAEAAACwNa4pAoLQ6dOnVVhYWOfUuc8//1zPPvusDh06pA4dOuiOO+7Q8OHDLUoJAAAQHDh9DghCc+bM0SeffKLXXnut5rljx45p7NixOnv2rBITE1VQUKAzZ87o+eef13/8x39YmBYAAODixulzQBD6+9//rhtvvLHWcxs3blRZWZl++9vf6g9/+IN27typK664QpmZmRalBAAACA6UIiAIHT9+XL1796713K5du5SUlKTU1FRJUqtWrfSzn/1MX3zxhRURAQAAggalCAhCDodDDoej5vHJkyd19OjROqfJderUSUVFRYGOBwAAEFQoRUAQiouL03vvvVfzeNeuXXI4HEpJSak17sSJE2rfvn2g4wEAAAQVVp8DgtDtt9+umTNnyuPxqEOHDvr973+vnj176pprrqk1bvfu3erTp49FKQEAAIIDpQgIQmPHjtXx48e1efNmeTwe9e/fX/Pnz1dIyL8/0qdOndKuXbs0bdo0C5MCAABc/FiSGwAAAICtcU0RAAAAAFujFAEAAACwNUoRAAAAAFujFAEAAACwNUoRAAAAAFujFAEAAACwNUoRAAAAAFujFAEAAACwtf8HfB787zOI3yIAAAAASUVORK5CYII="/>

`Parch`특성은 `SibSp`와 비슷하게 2명 이상의 부모나 자식과 함께 배에 탔을 때는 조금 더 생존했지만, 그렇지 않을 경우에는 생존한 사람의 비율이 적었다.



지금까지 살펴본 데이터 특성들을 간략하게 종합해보면,

성별이 여성일 수록(영화 타이타닉에서 나온 것 처럼 여성과 아이부터 먼저 살렸기 때문이 아닐까 싶고),

`Pclass`가 높을 수록(맨 위의 사진을 보면 타이타닉 호는 배의 후미부터 잠기기 시작되었다는 것을 알 수 있는데, 티켓의 등급이 높아질 수록 숙소가 배의 앞쪽과 위쪽으로 가는 경향이 있어 그 영향이 아닐까 싶고),

`Cherbourg` 선착장에서 배를 탔다면,

형제, 자매, 배우자, 부모, 자녀와 함께 배에 탔다면,

생존 확률이 더 높았다는 것을 볼 수 있다.



하지만 하나의 특성과 생존 비율 만을 생각해서 예측하기에는 무리가 있다.



예를 들어 높은 금액의 티켓(살 확률이 높은 숙소를 가진)을 산 부유한 사람이 가족들이랑 왔을 경우가 많다고 가정해본다면, 가족들과 함께 왔다고 해서 살 가능성이 높다고 할 수는 없으므로 단일 특성을 가지고 생존 확률을 예측하기보단 여러가지 특성을 종합해서 예측을 하는 것이 더 좋을 것이다.


# 4. 데이터 전처리 및 특성 추출


이제는 앞으로 예측할 모델에게 학습을 시킬 특성을 골라서 학습하기에 알맞게 전처리 과정을 진행 해볼 것이다.



의미를 찾지 못한 `Ticket`과 `Cabin` 특성을 제외한 나머지 특성을 가지고 전처리를 진행한다.



또한 데이터 전처리를 하는 과정에서는 훈련셋과 테스트셋을 같은 방법으로 한 번에 처리를 해야하므로 먼저 두 개의 데이터를 합쳐본다.



```python
train_and_test = [train, test]
```

## 4.1. 이름 특성


이름이 중요한 것 같이 않지만 `Name` 정보에는 Title이 있는데, 이를 통해서 승객의 성별이나 나이대, 결혼 유무를 알 수 있다. 성별과 나이는 이미 데이터에 들어 있지만 일단 Title을 가져오도록 한다.



데이터에 `Title`이라는 새로운 열을 만들어 Title 정보를 넣자.



```python
for dataset in train_and_test:
 	dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')

train.head(5)
```


  <div id="df-ae8d7f36-dd4a-4733-86ce-fd950837510f">
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ae8d7f36-dd4a-4733-86ce-fd950837510f')"
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
          document.querySelector('#df-ae8d7f36-dd4a-4733-86ce-fd950837510f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ae8d7f36-dd4a-4733-86ce-fd950837510f');
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
  


위에서 쓰인 ' ([A-Za-z]+)\.'는 정규표현식인데, 공백으로 시작하고, `.`로 끝나는 문자열을 추출할 때 저렇게 표현한다.



한편 추출한 Title을 가진 사람이 몇 명이나 존재하는지 성별과 함께 표현을 해보자.



```python
pd.crosstab(train['Title'], train['Sex'])
```


  <div id="df-d5c5a7b7-b497-4c29-b9e0-32ecc4a01ae8">
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
      <th>Sex</th>
      <th>female</th>
      <th>male</th>
    </tr>
    <tr>
      <th>Title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Capt</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Col</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Countess</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Don</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Dr</th>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Jonkheer</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Lady</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Major</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Master</th>
      <td>0</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Miss</th>
      <td>182</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mlle</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mme</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mr</th>
      <td>0</td>
      <td>517</td>
    </tr>
    <tr>
      <th>Mrs</th>
      <td>125</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Ms</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Rev</th>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Sir</th>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d5c5a7b7-b497-4c29-b9e0-32ecc4a01ae8')"
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
          document.querySelector('#df-d5c5a7b7-b497-4c29-b9e0-32ecc4a01ae8 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d5c5a7b7-b497-4c29-b9e0-32ecc4a01ae8');
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
  


여기서 흔하지 않은 Title은 Other로 대체하고 중복되는 표현을 통일하자.



```python
for dataset in train_and_test:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Don','Dona', 'Dr', 'Jonkheer',
                                                 'Lady','Major', 'Rev', 'Sir'], 'Other')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
```


  <div id="df-527d64d2-3bb6-4c7a-a6b7-a3465dc31d82">
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
      <th>Title</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Master</td>
      <td>0.575000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Miss</td>
      <td>0.702703</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mr</td>
      <td>0.156673</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mrs</td>
      <td>0.793651</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Other</td>
      <td>0.347826</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-527d64d2-3bb6-4c7a-a6b7-a3465dc31d82')"
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
          document.querySelector('#df-527d64d2-3bb6-4c7a-a6b7-a3465dc31d82 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-527d64d2-3bb6-4c7a-a6b7-a3465dc31d82');
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
  


그리고 추출한 Title 데이터를 학습하기 알맞게 String Data로 변형해주면 된다.



```python
for dataset in train_and_test:
    dataset['Title'] = dataset['Title'].astype(str)
```

## 4.2. 성 특성


이번에는 승객의 성별을 나타내는 `Sex` Feature를 처리할 것인데 이미 male과 female로 나뉘어져 있으므로 String Data로만 변형해주면 된다.



```python
for dataset in train_and_test:
    dataset['Sex'] = dataset['Sex'].astype(str)
```

## 4.3. 탑승 항구 특성


이제 배를 탑승한 선착장을 나타내는 `Embarked` Feature를 처리해보자.



일단 위에서 간략하게 살펴본 데이터 정보에 따르면 train 데이터에서 `Embarked` feature에는 NaN 값이 존재하며, 다음을 보면 잘 알 수 있다.



```python
train.isnull().sum()
```

<pre>
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
Title            0
dtype: int64
</pre>
`Embarked` 특성에 2개의 결측치를 확인할 수 있다.



데이터 분석시 결측치가 존재하면 안 되므로 이를 메꾸도록 한다.



```python
train['Embarked'].fillna('S',inplace=True)
```

여기서는 단순하게 이 두 사람은 사람이 제일 많이 탑승한 항구인 ‘Southampton’에서 탔다고 가정한다.


그리고 String Data로 변형



```python
for dataset in train_and_test:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].astype(str)
```

## 4.4. 나이 특성


`Age` Feature에도 NaN값은 존재하는데, 일단 빠진 값에는 나머지 모든 승객 나이의 평균을 넣어주자.



한편 연속적인 numeric data를 처리하는 방법에도 여러가지가 있는데, 이번에는 Binning을 사용할 것이다.



Binnig이란 여러 종류의 데이터에 대해 범위를 지정해주거나 카테고리를 통해 이전보다 작은 수의 그룹으로 만드는 기법이다.



이를 통해서 단일성 분포의 왜곡을 막을 수 있지만, 이산화를 통한 데이터의 손실이라는 단점도 존재한다.



이번에는 pd.cut()을 이용해 같은 길이의 구간을 가지는 다섯 개의 그룹을 만들어 보자.


이제 `Age`에 들어 있는 값을 위에서 구한 구간에 속하도록 바꿔준다.



```python
for dataset in train_and_test:
    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
    dataset['Age'] = dataset['Age'].astype(int)
    train['AgeBand'] = pd.cut(train['Age'], 5)
print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean()) # Survivied ratio about Age Band
```

<pre>
         AgeBand  Survived
0  (-0.08, 16.0]  0.550000
1   (16.0, 32.0]  0.344762
2   (32.0, 48.0]  0.403226
3   (48.0, 64.0]  0.434783
4   (64.0, 80.0]  0.090909
</pre>
이제 `Age`에 들어 있는 값을 위에서 구한 구간에 속하도록 바꿔준다.



```python
for dataset in train_and_test:
     dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
     dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
     dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
     dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
     dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
     dataset['Age'] = dataset['Age'].map( { 0: 'Child',  1: 'Young', 2: 'Middle', 3: 'Prime', 4: 'Old'} ).astype(str)
```

여기서 `Age`을 numeric이 아닌 string 형식으로 넣어주었는데, 숫자에 대한 경향성을 가지고 싶지 않아서 그렇게 했다.



사실 Binning과 같이 여기에도 장단점이 존재하는 것 같아 다음번에는 Numeric type으로 학습시켜서 어떻게 예측 결과가 달라지는지도 봐야겠다.


## 4.5. Fare 특성


Test 데이터 중에서 `Fare` Feature에도 NaN 값이 하나 존재하는데, Pclass와 Fare가 어느 정도 연관성이 있는 것 같아 Fare 데이터가 빠진 값의 Pclass를 가진 사람들의 평균 Fare를 넣어주는 식으로 처리를 해보자.



```python
print (train[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).mean())
print("")
print(test[test["Fare"].isnull()]["Pclass"])
```

<pre>
   Pclass       Fare
0       1  84.154687
1       2  20.662183
2       3  13.675550

152    3
Name: Pclass, dtype: int64
</pre>
위에서 볼 수 있듯이 누락된 데이터의 Pclass는 3이고, train 데이터에서 Pclass가 3인 사람들의 평균 Fare가 13.675550이므로 이 값을 넣어주자.



```python
for dataset in train_and_test:
    dataset['Fare'] = dataset['Fare'].fillna(13.675) # The only one empty fare data's pclass is 3.
```

`Age`에서 했던 것처럼 `Fare`에서도 Binning을 해보자. 이번에는 Age에서 했던 것 과는 다르게 Numeric한 값으로 남겨두자.



```python
for dataset in train_and_test:
    dataset.loc[ dataset['Fare'] <= 7.854, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare']   = 3
    dataset.loc[ dataset['Fare'] > 39.688, 'Fare'] = 4
    dataset['Fare'] = dataset['Fare'].astype(int)
```

## 4.6. 가족 특성


위에서 살펴봤듯이 형제, 자매, 배우자, 부모님, 자녀의 수가 많을 수록 생존한 경우가 많았는데, 두 개의 Feature를 합쳐서 `Family`라는 Feature로 만들자.



```python
for dataset in train_and_test:
    dataset["Family"] = dataset["Parch"] + dataset["SibSp"]
    dataset['Family'] = dataset['Family'].astype(int)
```

## 4.7. 특성 추출 및 나머지 전처리


이제 사용할 Feature에 대해서는 전처리가 되었으니, 학습시킬때 제외시킬 Feature들을 Drop 시키자.



```python
features_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'AgeBand'], axis=1)

print(train.head())
print(test.head())
```

<pre>
   Survived  Pclass     Sex     Age  Fare Embarked Title  Family
0         0       3    male   Young     0        S    Mr       1
1         1       1  female  Middle     4        C   Mrs       1
2         1       3  female   Young     1        S  Miss       0
3         1       1  female  Middle     4        S   Mrs       1
4         0       3    male  Middle     1        S    Mr       0
   PassengerId  Pclass     Sex     Age  Fare Embarked Title  Family
0          892       3    male  Middle     0        Q    Mr       0
1          893       3  female  Middle     0        S   Mrs       1
2          894       2    male   Prime     1        Q    Mr       0
3          895       3    male   Young     1        S    Mr       0
4          896       3  female   Young     2        S   Mrs       2
</pre>
위와 같이 가공된 train, test 데이터를 볼 수 있다.


마지막으로 Categorical Feature에 대해 one-hot encoding과 train data와 label을 분리시키는 작업을 하면 예측 모델에 학습시킬 준비가 끝났다.



```python
# One-hot-encoding for categorical variables
train = pd.get_dummies(train)
test = pd.get_dummies(test)

train_label = train['Survived']
train_data = train.drop('Survived', axis=1)
test_data = test.drop("PassengerId", axis=1).copy()
```

# 5. 모델 설계 및 학습


이번에 사용할 예측 모델은 다음과 같이 5가지가 있다.



1. Logistic Regression

2. Support Vector Machine (SVM)

3. k-Nearest Neighbor (kNN)

4. Random Forest

5. Naive Bayes



나중에 위의 모델에 대한 자세한 설명을 포스팅 할 텐데, 일단 이런 예측 모델이 있다고 하고 넘어가자.



일단 위 모델을 사용하기 위해서 필요한 `scikit-learn` 라이브러리를 불러오자.



```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.utils import shuffle
```

학습시키기 전에는 주어진 데이터가 정렬되어있어 학습에 방해가 될 수도 있으므로 섞어주도록 하자.



```python
train_data, train_label = shuffle(train_data, train_label, random_state = 5)
```

이제 모델 학습과 평가에 대한 pipeline을 만들자.



사실 scikit-learn에서 제공하는 fit()과 predict()를 사용하면 매우 간단하게 학습과 예측을 할 수 있어서 그냥 하나의 함수만 만들면 편하게 사용가능하다.



```python
def train_and_test(model):
    model.fit(train_data, train_label)
    prediction = model.predict(test_data)
    accuracy = round(model.score(train_data, train_label) * 100, 2)
    print("Accuracy : ", accuracy, "%")
    return prediction
```

이 함수에 다섯가지 모델을 넣어주면 학습과 평가가 완료된다.



```python
# Logistic Regression
log_pred = train_and_test(LogisticRegression())
# SVM
svm_pred = train_and_test(SVC())
#kNN
knn_pred_4 = train_and_test(KNeighborsClassifier(n_neighbors = 4))
# Random Forest
rf_pred = train_and_test(RandomForestClassifier(n_estimators=100))
# Navie Bayes
nb_pred = train_and_test(GaussianNB())
```

<pre>
Accuracy :  82.72 %
Accuracy :  83.5 %
Accuracy :  84.51 %
Accuracy :  88.55 %
Accuracy :  79.8 %
</pre>
# 6. 마무리


위에서 볼 수 있듯 4번째 모델인 Random Forest에서 가장 높은 정확도(88.55%)를 보였는데, 이 모델을 채택해서 submission 해보자.



```python
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": rf_pred
})

submission.to_csv('submission_rf.csv', index=False)
```

출처: https://cyc1am3n.github.io/2018/10/09/my-first-kaggle-competition_titanic.html 를 참고했음.

