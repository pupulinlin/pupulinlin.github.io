---
layout: single
title:  "4차 과제에 대한 내용입니다."
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


#
 
1
.
 
문
제
 
정
의
하
기


타
이
타
닉
호
에
 
탑
승
했
던
 
사
람
들
의
 
정
보
를
 
바
탕
으
로
 
생
존
자
를
 
예
측
하
는
 
문
제
이
다
.


#
 
2
.
 
데
이
터
 
불
러
오
기


먼
저
 
필
요
한
 
라
이
브
러
리
인
 
`
n
u
m
p
y
`
와
 
`
p
a
n
d
a
s
`
를
 
i
m
p
o
r
t
하
고
,
 
데
이
터
(
t
r
a
i
n
.
c
s
v
,
 
t
e
s
t
.
c
s
v
)
 
파
일
을
 
코
드
와
 
같
은
 
디
렉
토
리
에
 
다
운
을
 
받
고
 
p
d
.
r
e
a
d
_
c
s
v
 
를
 
이
용
해
서
 
불
러
오
자
.



```python
i
m
p
o
r
t
 
p
a
n
d
a
s
 
a
s
 
p
d

i
m
p
o
r
t
 
n
u
m
p
y
 
a
s
 
n
p


t
r
a
i
n
 
=
 
p
d
.
r
e
a
d
_
c
s
v
(
'
/
k
a
g
g
l
e
/
i
n
p
u
t
/
t
i
t
a
n
i
c
/
t
r
a
i
n
.
c
s
v
'
)

t
e
s
t
 
=
 
p
d
.
r
e
a
d
_
c
s
v
(
'
/
k
a
g
g
l
e
/
i
n
p
u
t
/
t
i
t
a
n
i
c
/
t
e
s
t
.
c
s
v
'
)
```

적
재
한
 
훈
련
데
이
터
를
 
확
인
하
기
 
위
해
 
h
e
a
d
(
)
 
메
서
드
를
 
이
용
하
여
 
앞
의
 
5
열
을
 
살
펴
본
다
.



```python
t
r
a
i
n
.
h
e
a
d
(
)
```

#
 
3
.
 
데
이
터
 
분
석


각
 
특
성
의
 
의
미
를
 
간
략
하
게
 
살
펴
보
면




*
 
S
u
r
v
i
v
i
e
d
는
 
생
존
 
여
부
(
0
은
 
사
망
,
 
1
은
 
생
존
;
 
t
r
a
i
n
 
데
이
터
에
서
만
 
제
공
)
,


*
 
P
c
l
a
s
s
는
 
사
회
경
제
적
 
지
위
(
1
에
 
가
까
울
 
수
록
 
높
음
)
,


*
 
S
i
p
S
p
는
 
배
우
자
나
 
형
제
 
자
매
 
명
 
수
의
 
총
 
합
,


*
 
P
a
r
c
h
는
 
부
모
 
자
식
 
명
 
수
의
 
총
 
합
을
 
나
타
낸
다
.




이
제
 
각
각
 
특
성
들
의
 
의
미
를
 
알
았
으
니
,
 
주
어
진
 
데
이
터
에
서
 
대
해
 
간
략
하
게
 
살
펴
보
자
.



```python
p
r
i
n
t
(
'
t
r
a
i
n
 
d
a
t
a
 
s
h
a
p
e
:
 
'
,
 
t
r
a
i
n
.
s
h
a
p
e
)

p
r
i
n
t
(
'
t
e
s
t
 
d
a
t
a
 
s
h
a
p
e
:
 
'
,
 
t
e
s
t
.
s
h
a
p
e
)

p
r
i
n
t
(
'
-
-
-
-
-
-
-
-
-
-
[
t
r
a
i
n
 
i
n
f
o
m
a
t
i
o
n
]
-
-
-
-
-
-
-
-
-
-
'
)

p
r
i
n
t
(
t
r
a
i
n
.
i
n
f
o
(
)
)

p
r
i
n
t
(
'
-
-
-
-
-
-
-
-
-
-
[
t
e
s
t
 
i
n
f
o
m
a
t
i
o
n
]
-
-
-
-
-
-
-
-
-
-
'
)

p
r
i
n
t
(
t
e
s
t
.
i
n
f
o
(
)
)
```

범
주
형
 
특
성
과
 
수
치
형
 
특
성
들
로
 
나
뉨
을
 
알
 
수
 
있
다
.


#
#
 
3
.
1
.
 
범
주
형
 
특
성
에
 
대
한
 
P
i
e
 
c
h
a
r
t


데
이
터
 
값
의
 
분
포
를
 
보
기
 
위
한
 
`
m
a
t
p
l
o
t
l
i
b
`
 
와
 
`
s
e
a
b
o
r
n
`
 
라
이
브
러
리
를
 
불
러
온
다
.



```python
```


```python
i
m
p
o
r
t
 
m
a
t
p
l
o
t
l
i
b
.
p
y
p
l
o
t
 
a
s
 
p
l
t

%
m
a
t
p
l
o
t
l
i
b
 
i
n
l
i
n
e

i
m
p
o
r
t
 
s
e
a
b
o
r
n
 
a
s
 
s
n
s

s
n
s
.
s
e
t
(
)
 
#
 
s
e
t
t
i
n
g
 
s
e
a
b
o
r
n
 
d
e
f
a
u
l
t
 
f
o
r
 
p
l
o
t
s
```

먼
저
 
다
음
과
 
같
은
 
범
주
형
 
특
성
의
 
분
포
를
 
보
기
 
위
해
서
 
P
i
e
 
c
h
a
r
t
를
 
만
드
는
 
함
수
를
 
정
의
해
보
자
.




*
 
S
e
x


*
 
P
c
l
a
s
s


*
 
E
m
b
a
r
k
e
d



```python
d
e
f
 
p
i
e
_
c
h
a
r
t
(
f
e
a
t
u
r
e
)
:

 
 
 
 
f
e
a
t
u
r
e
_
r
a
t
i
o
 
=
 
t
r
a
i
n
[
f
e
a
t
u
r
e
]
.
v
a
l
u
e
_
c
o
u
n
t
s
(
s
o
r
t
=
F
a
l
s
e
)

 
 
 
 
f
e
a
t
u
r
e
_
s
i
z
e
 
=
 
f
e
a
t
u
r
e
_
r
a
t
i
o
.
s
i
z
e

 
 
 
 
f
e
a
t
u
r
e
_
i
n
d
e
x
 
=
 
f
e
a
t
u
r
e
_
r
a
t
i
o
.
i
n
d
e
x

 
 
 
 
s
u
r
v
i
v
e
d
 
=
 
t
r
a
i
n
[
t
r
a
i
n
[
'
S
u
r
v
i
v
e
d
'
]
 
=
=
 
1
]
[
f
e
a
t
u
r
e
]
.
v
a
l
u
e
_
c
o
u
n
t
s
(
)

 
 
 
 
d
e
a
d
 
=
 
t
r
a
i
n
[
t
r
a
i
n
[
'
S
u
r
v
i
v
e
d
'
]
 
=
=
 
0
]
[
f
e
a
t
u
r
e
]
.
v
a
l
u
e
_
c
o
u
n
t
s
(
)

 


 
 
 
 
p
l
t
.
p
l
o
t
(
a
s
p
e
c
t
=
'
a
u
t
o
'
)

 
 
 
 
p
l
t
.
p
i
e
(
f
e
a
t
u
r
e
_
r
a
t
i
o
,
 
l
a
b
e
l
s
=
f
e
a
t
u
r
e
_
i
n
d
e
x
,
 
a
u
t
o
p
c
t
=
'
%
1
.
1
f
%
%
'
)

 
 
 
 
p
l
t
.
t
i
t
l
e
(
f
e
a
t
u
r
e
 
+
 
'
\
'
s
 
r
a
t
i
o
 
i
n
 
t
o
t
a
l
'
)

 
 
 
 
p
l
t
.
s
h
o
w
(
)


 
 
 
 
f
o
r
 
i
,
 
i
n
d
e
x
 
i
n
 
e
n
u
m
e
r
a
t
e
(
f
e
a
t
u
r
e
_
i
n
d
e
x
)
:

 
 
 
 
 
 
 
 
p
l
t
.
s
u
b
p
l
o
t
(
1
,
 
f
e
a
t
u
r
e
_
s
i
z
e
 
+
 
1
,
 
i
 
+
 
1
,
 
a
s
p
e
c
t
=
'
e
q
u
a
l
'
)

 
 
 
 
 
 
 
 
p
l
t
.
p
i
e
(
[
s
u
r
v
i
v
e
d
[
i
n
d
e
x
]
,
 
d
e
a
d
[
i
n
d
e
x
]
]
,
 
l
a
b
e
l
s
=
[
'
S
u
r
v
i
v
i
e
d
'
,
 
'
D
e
a
d
'
]
,
 
a
u
t
o
p
c
t
=
'
%
1
.
1
f
%
%
'
)

 
 
 
 
 
 
 
 
p
l
t
.
t
i
t
l
e
(
s
t
r
(
i
n
d
e
x
)
 
+
 
'
\
'
s
 
r
a
t
i
o
'
)


 
 
 
 
p
l
t
.
s
h
o
w
(
)
```

먼
저
 
`
S
e
x
`
에
 
대
해
서
 
P
i
e
 
c
h
a
r
t
를
 
그
려
보
면
,



```python
p
i
e
_
c
h
a
r
t
(
'
S
e
x
'
)
```

위
와
 
같
이
 
남
성
이
 
여
성
보
다
 
배
에
 
많
이
 
탔
으
며
,
 
남
성
보
다
 
여
성
의
 
생
존
 
비
율
이
 
높
다
는
 
것
을
 
알
 
수
가
 
있
다
.


이
제
 
사
회
경
제
적
 
지
위
인
 
`
P
c
l
a
s
s
`
에
 
대
해
서
도
 
그
려
보
자
.



```python
p
i
e
_
c
h
a
r
t
(
'
P
c
l
a
s
s
'
)
```

위
와
 
같
이
 
P
c
l
a
s
s
가
 
3
인
 
사
람
들
의
 
수
가
 
가
장
 
많
았
으
면
 
P
c
l
a
s
s
가
 
높
을
 
수
록
 
생
존
 
비
율
이
 
높
다
는
 
것
을
 
알
 
수
 
있
다
.


마
지
막
으
로
 
어
느
 
곳
에
서
 
배
를
 
탔
는
지
를
 
나
타
내
는
 
`
E
m
b
a
r
k
e
d
`
에
 
대
해
서
 
살
펴
보
자
.



```python
p
i
e
_
c
h
a
r
t
(
'
E
m
b
a
r
k
e
d
'
)
```

위
와
 
같
이
 
S
o
u
t
h
a
m
p
t
o
n
에
서
 
선
착
한
 
사
람
이
 
가
장
 
많
았
으
며
,
 
C
h
e
r
b
o
u
r
g
에
서
 
탄
 
사
람
 
중
에
 
생
존
한
 
사
람
의
 
비
율
이
 
높
았
고
,
 
나
머
지
 
두
 
선
착
장
에
서
 
탄
 
사
람
들
은
 
생
존
한
 
사
람
보
다
 
그
렇
지
 
못
한
 
사
람
이
 
조
금
 
더
 
많
았
다
.


#
#
 
3
.
2
.
 
범
주
형
 
특
성
에
 
대
한
 
B
a
r
 
c
h
a
r
t


이
번
에
는
 
아
래
의
 
특
성
들
에
 
대
허
서
 
B
a
r
 
c
h
a
r
t
를
 
정
의
해
서
 
데
이
터
를
 
시
각
화
 
해
보
자
.



```python
d
e
f
 
b
a
r
_
c
h
a
r
t
(
f
e
a
t
u
r
e
)
:

 
 
 
 
 
s
u
r
v
i
v
e
d
 
=
 
t
r
a
i
n
[
t
r
a
i
n
[
'
S
u
r
v
i
v
e
d
'
]
=
=
1
]
[
f
e
a
t
u
r
e
]
.
v
a
l
u
e
_
c
o
u
n
t
s
(
)

 
 
 
 
 
d
e
a
d
 
=
 
t
r
a
i
n
[
t
r
a
i
n
[
'
S
u
r
v
i
v
e
d
'
]
=
=
0
]
[
f
e
a
t
u
r
e
]
.
v
a
l
u
e
_
c
o
u
n
t
s
(
)

 
 
 
 
 
d
f
 
=
 
p
d
.
D
a
t
a
F
r
a
m
e
(
[
s
u
r
v
i
v
e
d
,
d
e
a
d
]
)

 
 
 
 
 
d
f
.
i
n
d
e
x
 
=
 
[
'
S
u
r
v
i
v
e
d
'
,
'
D
e
a
d
'
]

 
 
 
 
 
d
f
.
p
l
o
t
(
k
i
n
d
=
'
b
a
r
'
,
s
t
a
c
k
e
d
=
T
r
u
e
,
 
f
i
g
s
i
z
e
=
(
1
0
,
5
)
)
```

먼
저
`
S
i
b
S
p
`
에
 
대
해
서
 
B
a
r
 
c
h
a
r
t
를
 
그
려
보
자
.



```python
b
a
r
_
c
h
a
r
t
(
'
S
i
b
S
p
'
)
```

위
와
 
같
이
 
2
명
 
이
상
의
 
형
제
나
 
배
우
자
와
 
함
께
 
탔
을
 
경
우
 
생
존
한
 
사
람
의
 
비
율
이
 
컸
다
는
 
것
을
 
볼
 
수
 
있
고
,
 
그
렇
지
 
않
을
 
경
우
에
는
 
생
존
한
 
사
람
의
 
비
율
이
 
적
었
다
는
 
것
을
 
볼
 
수
 
있
다
.



```python
b
a
r
_
c
h
a
r
t
(
'
P
a
r
c
h
'
)
```

`
P
a
r
c
h
`
특
성
은
 
`
S
i
b
S
p
`
와
 
비
슷
하
게
 
2
명
 
이
상
의
 
부
모
나
 
자
식
과
 
함
께
 
배
에
 
탔
을
 
때
는
 
조
금
 
더
 
생
존
했
지
만
,
 
그
렇
지
 
않
을
 
경
우
에
는
 
생
존
한
 
사
람
의
 
비
율
이
 
적
었
다
.




지
금
까
지
 
살
펴
본
 
데
이
터
 
특
성
들
을
 
간
략
하
게
 
종
합
해
보
면
,


성
별
이
 
여
성
일
 
수
록
(
영
화
 
타
이
타
닉
에
서
 
나
온
 
것
 
처
럼
 
여
성
과
 
아
이
부
터
 
먼
저
 
살
렸
기
 
때
문
이
 
아
닐
까
 
싶
고
)
,


`
P
c
l
a
s
s
`
가
 
높
을
 
수
록
(
맨
 
위
의
 
사
진
을
 
보
면
 
타
이
타
닉
 
호
는
 
배
의
 
후
미
부
터
 
잠
기
기
 
시
작
되
었
다
는
 
것
을
 
알
 
수
 
있
는
데
,
 
티
켓
의
 
등
급
이
 
높
아
질
 
수
록
 
숙
소
가
 
배
의
 
앞
쪽
과
 
위
쪽
으
로
 
가
는
 
경
향
이
 
있
어
 
그
 
영
향
이
 
아
닐
까
 
싶
고
)
,


`
C
h
e
r
b
o
u
r
g
`
 
선
착
장
에
서
 
배
를
 
탔
다
면
,


형
제
,
 
자
매
,
 
배
우
자
,
 
부
모
,
 
자
녀
와
 
함
께
 
배
에
 
탔
다
면
,


생
존
 
확
률
이
 
더
 
높
았
다
는
 
것
을
 
볼
 
수
 
있
다
.




하
지
만
 
하
나
의
 
특
성
과
 
생
존
 
비
율
 
만
을
 
생
각
해
서
 
예
측
하
기
에
는
 
무
리
가
 
있
다
.




예
를
 
들
어
 
높
은
 
금
액
의
 
티
켓
(
살
 
확
률
이
 
높
은
 
숙
소
를
 
가
진
)
을
 
산
 
부
유
한
 
사
람
이
 
가
족
들
이
랑
 
왔
을
 
경
우
가
 
많
다
고
 
가
정
해
본
다
면
,
 
가
족
들
과
 
함
께
 
왔
다
고
 
해
서
 
살
 
가
능
성
이
 
높
다
고
 
할
 
수
는
 
없
으
므
로
 
단
일
 
특
성
을
 
가
지
고
 
생
존
 
확
률
을
 
예
측
하
기
보
단
 
여
러
가
지
 
특
성
을
 
종
합
해
서
 
예
측
을
 
하
는
 
것
이
 
더
 
좋
을
 
것
이
다
.


#
 
4
.
 
데
이
터
 
전
처
리
 
및
 
특
성
 
추
출


이
제
는
 
앞
으
로
 
예
측
할
 
모
델
에
게
 
학
습
을
 
시
킬
 
특
성
을
 
골
라
서
 
학
습
하
기
에
 
알
맞
게
 
전
처
리
 
과
정
을
 
진
행
 
해
볼
 
것
이
다
.




의
미
를
 
찾
지
 
못
한
 
`
T
i
c
k
e
t
`
과
 
`
C
a
b
i
n
`
 
특
성
을
 
제
외
한
 
나
머
지
 
특
성
을
 
가
지
고
 
전
처
리
를
 
진
행
한
다
.




또
한
 
데
이
터
 
전
처
리
를
 
하
는
 
과
정
에
서
는
 
훈
련
셋
과
 
테
스
트
셋
을
 
같
은
 
방
법
으
로
 
한
 
번
에
 
처
리
를
 
해
야
하
므
로
 
먼
저
 
두
 
개
의
 
데
이
터
를
 
합
쳐
본
다
.



```python
t
r
a
i
n
_
a
n
d
_
t
e
s
t
 
=
 
[
t
r
a
i
n
,
 
t
e
s
t
]
```

#
#
 
4
.
1
.
 
이
름
 
특
성


이
름
이
 
중
요
한
 
것
 
같
이
 
않
지
만
 
`
N
a
m
e
`
 
정
보
에
는
 
T
i
t
l
e
이
 
있
는
데
,
 
이
를
 
통
해
서
 
승
객
의
 
성
별
이
나
 
나
이
대
,
 
결
혼
 
유
무
를
 
알
 
수
 
있
다
.
 
성
별
과
 
나
이
는
 
이
미
 
데
이
터
에
 
들
어
 
있
지
만
 
일
단
 
T
i
t
l
e
을
 
가
져
오
도
록
 
한
다
.




데
이
터
에
 
`
T
i
t
l
e
`
이
라
는
 
새
로
운
 
열
을
 
만
들
어
 
T
i
t
l
e
 
정
보
를
 
넣
자
.



```python
f
o
r
 
d
a
t
a
s
e
t
 
i
n
 
t
r
a
i
n
_
a
n
d
_
t
e
s
t
:

 
	
d
a
t
a
s
e
t
[
'
T
i
t
l
e
'
]
 
=
 
d
a
t
a
s
e
t
.
N
a
m
e
.
s
t
r
.
e
x
t
r
a
c
t
(
'
 
(
[
A
-
Z
a
-
z
]
+
)
\
.
'
)


t
r
a
i
n
.
h
e
a
d
(
5
)
```

위
에
서
 
쓰
인
 
'
 
(
[
A
-
Z
a
-
z
]
+
)
\
.
'
는
 
정
규
표
현
식
인
데
,
 
공
백
으
로
 
시
작
하
고
,
 
`
.
`
로
 
끝
나
는
 
문
자
열
을
 
추
출
할
 
때
 
저
렇
게
 
표
현
한
다
.




한
편
 
추
출
한
 
T
i
t
l
e
을
 
가
진
 
사
람
이
 
몇
 
명
이
나
 
존
재
하
는
지
 
성
별
과
 
함
께
 
표
현
을
 
해
보
자
.



```python
p
d
.
c
r
o
s
s
t
a
b
(
t
r
a
i
n
[
'
T
i
t
l
e
'
]
,
 
t
r
a
i
n
[
'
S
e
x
'
]
)
```

여
기
서
 
흔
하
지
 
않
은
 
T
i
t
l
e
은
 
O
t
h
e
r
로
 
대
체
하
고
 
중
복
되
는
 
표
현
을
 
통
일
하
자
.



```python
f
o
r
 
d
a
t
a
s
e
t
 
i
n
 
t
r
a
i
n
_
a
n
d
_
t
e
s
t
:

 
 
 
 
d
a
t
a
s
e
t
[
'
T
i
t
l
e
'
]
 
=
 
d
a
t
a
s
e
t
[
'
T
i
t
l
e
'
]
.
r
e
p
l
a
c
e
(
[
'
C
a
p
t
'
,
 
'
C
o
l
'
,
 
'
C
o
u
n
t
e
s
s
'
,
 
'
D
o
n
'
,
'
D
o
n
a
'
,
 
'
D
r
'
,
 
'
J
o
n
k
h
e
e
r
'
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
L
a
d
y
'
,
'
M
a
j
o
r
'
,
 
'
R
e
v
'
,
 
'
S
i
r
'
]
,
 
'
O
t
h
e
r
'
)

 
 
 
 
d
a
t
a
s
e
t
[
'
T
i
t
l
e
'
]
 
=
 
d
a
t
a
s
e
t
[
'
T
i
t
l
e
'
]
.
r
e
p
l
a
c
e
(
'
M
l
l
e
'
,
 
'
M
i
s
s
'
)

 
 
 
 
d
a
t
a
s
e
t
[
'
T
i
t
l
e
'
]
 
=
 
d
a
t
a
s
e
t
[
'
T
i
t
l
e
'
]
.
r
e
p
l
a
c
e
(
'
M
m
e
'
,
 
'
M
r
s
'
)

 
 
 
 
d
a
t
a
s
e
t
[
'
T
i
t
l
e
'
]
 
=
 
d
a
t
a
s
e
t
[
'
T
i
t
l
e
'
]
.
r
e
p
l
a
c
e
(
'
M
s
'
,
 
'
M
i
s
s
'
)


t
r
a
i
n
[
[
'
T
i
t
l
e
'
,
 
'
S
u
r
v
i
v
e
d
'
]
]
.
g
r
o
u
p
b
y
(
[
'
T
i
t
l
e
'
]
,
 
a
s
_
i
n
d
e
x
=
F
a
l
s
e
)
.
m
e
a
n
(
)
```

그
리
고
 
추
출
한
 
T
i
t
l
e
 
데
이
터
를
 
학
습
하
기
 
알
맞
게
 
S
t
r
i
n
g
 
D
a
t
a
로
 
변
형
해
주
면
 
된
다
.



```python
f
o
r
 
d
a
t
a
s
e
t
 
i
n
 
t
r
a
i
n
_
a
n
d
_
t
e
s
t
:

 
 
 
 
d
a
t
a
s
e
t
[
'
T
i
t
l
e
'
]
 
=
 
d
a
t
a
s
e
t
[
'
T
i
t
l
e
'
]
.
a
s
t
y
p
e
(
s
t
r
)
```

#
#
 
4
.
2
.
 
성
 
특
성


이
번
에
는
 
승
객
의
 
성
별
을
 
나
타
내
는
 
`
S
e
x
`
 
F
e
a
t
u
r
e
를
 
처
리
할
 
것
인
데
 
이
미
 
m
a
l
e
과
 
f
e
m
a
l
e
로
 
나
뉘
어
져
 
있
으
므
로
 
S
t
r
i
n
g
 
D
a
t
a
로
만
 
변
형
해
주
면
 
된
다
.



```python
f
o
r
 
d
a
t
a
s
e
t
 
i
n
 
t
r
a
i
n
_
a
n
d
_
t
e
s
t
:

 
 
 
 
d
a
t
a
s
e
t
[
'
S
e
x
'
]
 
=
 
d
a
t
a
s
e
t
[
'
S
e
x
'
]
.
a
s
t
y
p
e
(
s
t
r
)
```

#
#
 
4
.
3
.
 
탑
승
 
항
구
 
특
성


이
제
 
배
를
 
탑
승
한
 
선
착
장
을
 
나
타
내
는
 
`
E
m
b
a
r
k
e
d
`
 
F
e
a
t
u
r
e
를
 
처
리
해
보
자
.




일
단
 
위
에
서
 
간
략
하
게
 
살
펴
본
 
데
이
터
 
정
보
에
 
따
르
면
 
t
r
a
i
n
 
데
이
터
에
서
 
`
E
m
b
a
r
k
e
d
`
 
f
e
a
t
u
r
e
에
는
 
N
a
N
 
값
이
 
존
재
하
며
,
 
다
음
을
 
보
면
 
잘
 
알
 
수
 
있
다
.



```python
t
r
a
i
n
.
i
s
n
u
l
l
(
)
.
s
u
m
(
)
```

`
E
m
b
a
r
k
e
d
`
 
특
성
에
 
2
개
의
 
결
측
치
를
 
확
인
할
 
수
 
있
다
.




데
이
터
 
분
석
시
 
결
측
치
가
 
존
재
하
면
 
안
 
되
므
로
 
이
를
 
메
꾸
도
록
 
한
다
.



```python
t
r
a
i
n
[
'
E
m
b
a
r
k
e
d
'
]
.
f
i
l
l
n
a
(
'
S
'
,
i
n
p
l
a
c
e
=
T
r
u
e
)
```

여
기
서
는
 
단
순
하
게
 
이
 
두
 
사
람
은
 
사
람
이
 
제
일
 
많
이
 
탑
승
한
 
항
구
인
 
‘
S
o
u
t
h
a
m
p
t
o
n
’
에
서
 
탔
다
고
 
가
정
한
다
.


그
리
고
 
S
t
r
i
n
g
 
D
a
t
a
로
 
변
형



```python
f
o
r
 
d
a
t
a
s
e
t
 
i
n
 
t
r
a
i
n
_
a
n
d
_
t
e
s
t
:

 
 
 
 
d
a
t
a
s
e
t
[
'
E
m
b
a
r
k
e
d
'
]
 
=
 
d
a
t
a
s
e
t
[
'
E
m
b
a
r
k
e
d
'
]
.
f
i
l
l
n
a
(
'
S
'
)

 
 
 
 
d
a
t
a
s
e
t
[
'
E
m
b
a
r
k
e
d
'
]
 
=
 
d
a
t
a
s
e
t
[
'
E
m
b
a
r
k
e
d
'
]
.
a
s
t
y
p
e
(
s
t
r
)
```

#
#
 
4
.
4
.
 
나
이
 
특
성


`
A
g
e
`
 
F
e
a
t
u
r
e
에
도
 
N
a
N
값
은
 
존
재
하
는
데
,
 
일
단
 
빠
진
 
값
에
는
 
나
머
지
 
모
든
 
승
객
 
나
이
의
 
평
균
을
 
넣
어
주
자
.




한
편
 
연
속
적
인
 
n
u
m
e
r
i
c
 
d
a
t
a
를
 
처
리
하
는
 
방
법
에
도
 
여
러
가
지
가
 
있
는
데
,
 
이
번
에
는
 
B
i
n
n
i
n
g
을
 
사
용
할
 
것
이
다
.




B
i
n
n
i
g
이
란
 
여
러
 
종
류
의
 
데
이
터
에
 
대
해
 
범
위
를
 
지
정
해
주
거
나
 
카
테
고
리
를
 
통
해
 
이
전
보
다
 
작
은
 
수
의
 
그
룹
으
로
 
만
드
는
 
기
법
이
다
.




이
를
 
통
해
서
 
단
일
성
 
분
포
의
 
왜
곡
을
 
막
을
 
수
 
있
지
만
,
 
이
산
화
를
 
통
한
 
데
이
터
의
 
손
실
이
라
는
 
단
점
도
 
존
재
한
다
.




이
번
에
는
 
p
d
.
c
u
t
(
)
을
 
이
용
해
 
같
은
 
길
이
의
 
구
간
을
 
가
지
는
 
다
섯
 
개
의
 
그
룹
을
 
만
들
어
 
보
자
.


이
제
 
`
A
g
e
`
에
 
들
어
 
있
는
 
값
을
 
위
에
서
 
구
한
 
구
간
에
 
속
하
도
록
 
바
꿔
준
다
.



```python
f
o
r
 
d
a
t
a
s
e
t
 
i
n
 
t
r
a
i
n
_
a
n
d
_
t
e
s
t
:

 
 
 
 
d
a
t
a
s
e
t
[
'
A
g
e
'
]
.
f
i
l
l
n
a
(
d
a
t
a
s
e
t
[
'
A
g
e
'
]
.
m
e
a
n
(
)
,
 
i
n
p
l
a
c
e
=
T
r
u
e
)

 
 
 
 
d
a
t
a
s
e
t
[
'
A
g
e
'
]
 
=
 
d
a
t
a
s
e
t
[
'
A
g
e
'
]
.
a
s
t
y
p
e
(
i
n
t
)

 
 
 
 
t
r
a
i
n
[
'
A
g
e
B
a
n
d
'
]
 
=
 
p
d
.
c
u
t
(
t
r
a
i
n
[
'
A
g
e
'
]
,
 
5
)

p
r
i
n
t
 
(
t
r
a
i
n
[
[
'
A
g
e
B
a
n
d
'
,
 
'
S
u
r
v
i
v
e
d
'
]
]
.
g
r
o
u
p
b
y
(
[
'
A
g
e
B
a
n
d
'
]
,
 
a
s
_
i
n
d
e
x
=
F
a
l
s
e
)
.
m
e
a
n
(
)
)
 
#
 
S
u
r
v
i
v
i
e
d
 
r
a
t
i
o
 
a
b
o
u
t
 
A
g
e
 
B
a
n
d
```

이
제
 
`
A
g
e
`
에
 
들
어
 
있
는
 
값
을
 
위
에
서
 
구
한
 
구
간
에
 
속
하
도
록
 
바
꿔
준
다
.



```python
f
o
r
 
d
a
t
a
s
e
t
 
i
n
 
t
r
a
i
n
_
a
n
d
_
t
e
s
t
:

 
 
 
 
 
d
a
t
a
s
e
t
.
l
o
c
[
 
d
a
t
a
s
e
t
[
'
A
g
e
'
]
 
<
=
 
1
6
,
 
'
A
g
e
'
]
 
=
 
0

 
 
 
 
 
d
a
t
a
s
e
t
.
l
o
c
[
(
d
a
t
a
s
e
t
[
'
A
g
e
'
]
 
>
 
1
6
)
 
&
 
(
d
a
t
a
s
e
t
[
'
A
g
e
'
]
 
<
=
 
3
2
)
,
 
'
A
g
e
'
]
 
=
 
1

 
 
 
 
 
d
a
t
a
s
e
t
.
l
o
c
[
(
d
a
t
a
s
e
t
[
'
A
g
e
'
]
 
>
 
3
2
)
 
&
 
(
d
a
t
a
s
e
t
[
'
A
g
e
'
]
 
<
=
 
4
8
)
,
 
'
A
g
e
'
]
 
=
 
2

 
 
 
 
 
d
a
t
a
s
e
t
.
l
o
c
[
(
d
a
t
a
s
e
t
[
'
A
g
e
'
]
 
>
 
4
8
)
 
&
 
(
d
a
t
a
s
e
t
[
'
A
g
e
'
]
 
<
=
 
6
4
)
,
 
'
A
g
e
'
]
 
=
 
3

 
 
 
 
 
d
a
t
a
s
e
t
.
l
o
c
[
 
d
a
t
a
s
e
t
[
'
A
g
e
'
]
 
>
 
6
4
,
 
'
A
g
e
'
]
 
=
 
4

 
 
 
 
 
d
a
t
a
s
e
t
[
'
A
g
e
'
]
 
=
 
d
a
t
a
s
e
t
[
'
A
g
e
'
]
.
m
a
p
(
 
{
 
0
:
 
'
C
h
i
l
d
'
,
 
 
1
:
 
'
Y
o
u
n
g
'
,
 
2
:
 
'
M
i
d
d
l
e
'
,
 
3
:
 
'
P
r
i
m
e
'
,
 
4
:
 
'
O
l
d
'
}
 
)
.
a
s
t
y
p
e
(
s
t
r
)
```

여
기
서
 
`
A
g
e
`
을
 
n
u
m
e
r
i
c
이
 
아
닌
 
s
t
r
i
n
g
 
형
식
으
로
 
넣
어
주
었
는
데
,
 
숫
자
에
 
대
한
 
경
향
성
을
 
가
지
고
 
싶
지
 
않
아
서
 
그
렇
게
 
했
다
.




사
실
 
B
i
n
n
i
n
g
과
 
같
이
 
여
기
에
도
 
장
단
점
이
 
존
재
하
는
 
것
 
같
아
 
다
음
번
에
는
 
N
u
m
e
r
i
c
 
t
y
p
e
으
로
 
학
습
시
켜
서
 
어
떻
게
 
예
측
 
결
과
가
 
달
라
지
는
지
도
 
봐
야
겠
다
.


#
#
 
4
.
5
.
 
F
a
r
e
 
특
성


T
e
s
t
 
데
이
터
 
중
에
서
 
`
F
a
r
e
`
 
F
e
a
t
u
r
e
에
도
 
N
a
N
 
값
이
 
하
나
 
존
재
하
는
데
,
 
P
c
l
a
s
s
와
 
F
a
r
e
가
 
어
느
 
정
도
 
연
관
성
이
 
있
는
 
것
 
같
아
 
F
a
r
e
 
데
이
터
가
 
빠
진
 
값
의
 
P
c
l
a
s
s
를
 
가
진
 
사
람
들
의
 
평
균
 
F
a
r
e
를
 
넣
어
주
는
 
식
으
로
 
처
리
를
 
해
보
자
.



```python
p
r
i
n
t
 
(
t
r
a
i
n
[
[
'
P
c
l
a
s
s
'
,
 
'
F
a
r
e
'
]
]
.
g
r
o
u
p
b
y
(
[
'
P
c
l
a
s
s
'
]
,
 
a
s
_
i
n
d
e
x
=
F
a
l
s
e
)
.
m
e
a
n
(
)
)

p
r
i
n
t
(
"
"
)

p
r
i
n
t
(
t
e
s
t
[
t
e
s
t
[
"
F
a
r
e
"
]
.
i
s
n
u
l
l
(
)
]
[
"
P
c
l
a
s
s
"
]
)
```

위
에
서
 
볼
 
수
 
있
듯
이
 
누
락
된
 
데
이
터
의
 
P
c
l
a
s
s
는
 
3
이
고
,
 
t
r
a
i
n
 
데
이
터
에
서
 
P
c
l
a
s
s
가
 
3
인
 
사
람
들
의
 
평
균
 
F
a
r
e
가
 
1
3
.
6
7
5
5
5
0
이
므
로
 
이
 
값
을
 
넣
어
주
자
.



```python
f
o
r
 
d
a
t
a
s
e
t
 
i
n
 
t
r
a
i
n
_
a
n
d
_
t
e
s
t
:

 
 
 
 
d
a
t
a
s
e
t
[
'
F
a
r
e
'
]
 
=
 
d
a
t
a
s
e
t
[
'
F
a
r
e
'
]
.
f
i
l
l
n
a
(
1
3
.
6
7
5
)
 
#
 
T
h
e
 
o
n
l
y
 
o
n
e
 
e
m
p
t
y
 
f
a
r
e
 
d
a
t
a
'
s
 
p
c
l
a
s
s
 
i
s
 
3
.
```

`
A
g
e
`
에
서
 
했
던
 
것
처
럼
 
`
F
a
r
e
`
에
서
도
 
B
i
n
n
i
n
g
을
 
해
보
자
.
 
이
번
에
는
 
A
g
e
에
서
 
했
던
 
것
 
과
는
 
다
르
게
 
N
u
m
e
r
i
c
한
 
값
으
로
 
남
겨
두
자
.



```python
f
o
r
 
d
a
t
a
s
e
t
 
i
n
 
t
r
a
i
n
_
a
n
d
_
t
e
s
t
:

 
 
 
 
d
a
t
a
s
e
t
.
l
o
c
[
 
d
a
t
a
s
e
t
[
'
F
a
r
e
'
]
 
<
=
 
7
.
8
5
4
,
 
'
F
a
r
e
'
]
 
=
 
0

 
 
 
 
d
a
t
a
s
e
t
.
l
o
c
[
(
d
a
t
a
s
e
t
[
'
F
a
r
e
'
]
 
>
 
7
.
8
5
4
)
 
&
 
(
d
a
t
a
s
e
t
[
'
F
a
r
e
'
]
 
<
=
 
1
0
.
5
)
,
 
'
F
a
r
e
'
]
 
=
 
1

 
 
 
 
d
a
t
a
s
e
t
.
l
o
c
[
(
d
a
t
a
s
e
t
[
'
F
a
r
e
'
]
 
>
 
1
0
.
5
)
 
&
 
(
d
a
t
a
s
e
t
[
'
F
a
r
e
'
]
 
<
=
 
2
1
.
6
7
9
)
,
 
'
F
a
r
e
'
]
 
 
 
=
 
2

 
 
 
 
d
a
t
a
s
e
t
.
l
o
c
[
(
d
a
t
a
s
e
t
[
'
F
a
r
e
'
]
 
>
 
2
1
.
6
7
9
)
 
&
 
(
d
a
t
a
s
e
t
[
'
F
a
r
e
'
]
 
<
=
 
3
9
.
6
8
8
)
,
 
'
F
a
r
e
'
]
 
 
 
=
 
3

 
 
 
 
d
a
t
a
s
e
t
.
l
o
c
[
 
d
a
t
a
s
e
t
[
'
F
a
r
e
'
]
 
>
 
3
9
.
6
8
8
,
 
'
F
a
r
e
'
]
 
=
 
4

 
 
 
 
d
a
t
a
s
e
t
[
'
F
a
r
e
'
]
 
=
 
d
a
t
a
s
e
t
[
'
F
a
r
e
'
]
.
a
s
t
y
p
e
(
i
n
t
)
```

#
#
 
4
.
6
.
 
가
족
 
특
성


위
에
서
 
살
펴
봤
듯
이
 
형
제
,
 
자
매
,
 
배
우
자
,
 
부
모
님
,
 
자
녀
의
 
수
가
 
많
을
 
수
록
 
생
존
한
 
경
우
가
 
많
았
는
데
,
 
두
 
개
의
 
F
e
a
t
u
r
e
를
 
합
쳐
서
 
`
F
a
m
i
l
y
`
라
는
 
F
e
a
t
u
r
e
로
 
만
들
자
.



```python
f
o
r
 
d
a
t
a
s
e
t
 
i
n
 
t
r
a
i
n
_
a
n
d
_
t
e
s
t
:

 
 
 
 
d
a
t
a
s
e
t
[
"
F
a
m
i
l
y
"
]
 
=
 
d
a
t
a
s
e
t
[
"
P
a
r
c
h
"
]
 
+
 
d
a
t
a
s
e
t
[
"
S
i
b
S
p
"
]

 
 
 
 
d
a
t
a
s
e
t
[
'
F
a
m
i
l
y
'
]
 
=
 
d
a
t
a
s
e
t
[
'
F
a
m
i
l
y
'
]
.
a
s
t
y
p
e
(
i
n
t
)
```

#
#
 
4
.
7
.
 
특
성
 
추
출
 
및
 
나
머
지
 
전
처
리


이
제
 
사
용
할
 
F
e
a
t
u
r
e
에
 
대
해
서
는
 
전
처
리
가
 
되
었
으
니
,
 
학
습
시
킬
때
 
제
외
시
킬
 
F
e
a
t
u
r
e
들
을
 
D
r
o
p
 
시
키
자
.



```python
f
e
a
t
u
r
e
s
_
d
r
o
p
 
=
 
[
'
N
a
m
e
'
,
 
'
T
i
c
k
e
t
'
,
 
'
C
a
b
i
n
'
,
 
'
S
i
b
S
p
'
,
 
'
P
a
r
c
h
'
]

t
r
a
i
n
 
=
 
t
r
a
i
n
.
d
r
o
p
(
f
e
a
t
u
r
e
s
_
d
r
o
p
,
 
a
x
i
s
=
1
)

t
e
s
t
 
=
 
t
e
s
t
.
d
r
o
p
(
f
e
a
t
u
r
e
s
_
d
r
o
p
,
 
a
x
i
s
=
1
)

t
r
a
i
n
 
=
 
t
r
a
i
n
.
d
r
o
p
(
[
'
P
a
s
s
e
n
g
e
r
I
d
'
,
 
'
A
g
e
B
a
n
d
'
]
,
 
a
x
i
s
=
1
)


p
r
i
n
t
(
t
r
a
i
n
.
h
e
a
d
(
)
)

p
r
i
n
t
(
t
e
s
t
.
h
e
a
d
(
)
)
```

위
와
 
같
이
 
가
공
된
 
t
r
a
i
n
,
 
t
e
s
t
 
데
이
터
를
 
볼
 
수
 
있
다
.


마
지
막
으
로
 
C
a
t
e
g
o
r
i
c
a
l
 
F
e
a
t
u
r
e
에
 
대
해
 
o
n
e
-
h
o
t
 
e
n
c
o
d
i
n
g
과
 
t
r
a
i
n
 
d
a
t
a
와
 
l
a
b
e
l
을
 
분
리
시
키
는
 
작
업
을
 
하
면
 
예
측
 
모
델
에
 
학
습
시
킬
 
준
비
가
 
끝
났
다
.



```python
#
 
O
n
e
-
h
o
t
-
e
n
c
o
d
i
n
g
 
f
o
r
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s

t
r
a
i
n
 
=
 
p
d
.
g
e
t
_
d
u
m
m
i
e
s
(
t
r
a
i
n
)

t
e
s
t
 
=
 
p
d
.
g
e
t
_
d
u
m
m
i
e
s
(
t
e
s
t
)


t
r
a
i
n
_
l
a
b
e
l
 
=
 
t
r
a
i
n
[
'
S
u
r
v
i
v
e
d
'
]

t
r
a
i
n
_
d
a
t
a
 
=
 
t
r
a
i
n
.
d
r
o
p
(
'
S
u
r
v
i
v
e
d
'
,
 
a
x
i
s
=
1
)

t
e
s
t
_
d
a
t
a
 
=
 
t
e
s
t
.
d
r
o
p
(
"
P
a
s
s
e
n
g
e
r
I
d
"
,
 
a
x
i
s
=
1
)
.
c
o
p
y
(
)
```

#
 
5
.
 
모
델
 
설
계
 
및
 
학
습


이
번
에
 
사
용
할
 
예
측
 
모
델
은
 
다
음
과
 
같
이
 
5
가
지
가
 
있
다
.




1
.
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n


2
.
 
S
u
p
p
o
r
t
 
V
e
c
t
o
r
 
M
a
c
h
i
n
e
 
(
S
V
M
)


3
.
 
k
-
N
e
a
r
e
s
t
 
N
e
i
g
h
b
o
r
 
(
k
N
N
)


4
.
 
R
a
n
d
o
m
 
F
o
r
e
s
t


5
.
 
N
a
i
v
e
 
B
a
y
e
s




나
중
에
 
위
의
 
모
델
에
 
대
한
 
자
세
한
 
설
명
을
 
포
스
팅
 
할
 
텐
데
,
 
일
단
 
이
런
 
예
측
 
모
델
이
 
있
다
고
 
하
고
 
넘
어
가
자
.




일
단
 
위
 
모
델
을
 
사
용
하
기
 
위
해
서
 
필
요
한
 
`
s
c
i
k
i
t
-
l
e
a
r
n
`
 
라
이
브
러
리
를
 
불
러
오
자
.



```python
f
r
o
m
 
s
k
l
e
a
r
n
.
l
i
n
e
a
r
_
m
o
d
e
l
 
i
m
p
o
r
t
 
L
o
g
i
s
t
i
c
R
e
g
r
e
s
s
i
o
n

f
r
o
m
 
s
k
l
e
a
r
n
.
s
v
m
 
i
m
p
o
r
t
 
S
V
C

f
r
o
m
 
s
k
l
e
a
r
n
.
n
e
i
g
h
b
o
r
s
 
i
m
p
o
r
t
 
K
N
e
i
g
h
b
o
r
s
C
l
a
s
s
i
f
i
e
r

f
r
o
m
 
s
k
l
e
a
r
n
.
e
n
s
e
m
b
l
e
 
i
m
p
o
r
t
 
R
a
n
d
o
m
F
o
r
e
s
t
C
l
a
s
s
i
f
i
e
r

f
r
o
m
 
s
k
l
e
a
r
n
.
n
a
i
v
e
_
b
a
y
e
s
 
i
m
p
o
r
t
 
G
a
u
s
s
i
a
n
N
B


f
r
o
m
 
s
k
l
e
a
r
n
.
u
t
i
l
s
 
i
m
p
o
r
t
 
s
h
u
f
f
l
e
```

학
습
시
키
기
 
전
에
는
 
주
어
진
 
데
이
터
가
 
정
렬
되
어
있
어
 
학
습
에
 
방
해
가
 
될
 
수
도
 
있
으
므
로
 
섞
어
주
도
록
 
하
자
.



```python
t
r
a
i
n
_
d
a
t
a
,
 
t
r
a
i
n
_
l
a
b
e
l
 
=
 
s
h
u
f
f
l
e
(
t
r
a
i
n
_
d
a
t
a
,
 
t
r
a
i
n
_
l
a
b
e
l
,
 
r
a
n
d
o
m
_
s
t
a
t
e
 
=
 
5
)
```

이
제
 
모
델
 
학
습
과
 
평
가
에
 
대
한
 
p
i
p
e
l
i
n
e
을
 
만
들
자
.




사
실
 
s
c
i
k
i
t
-
l
e
a
r
n
에
서
 
제
공
하
는
 
f
i
t
(
)
과
 
p
r
e
d
i
c
t
(
)
를
 
사
용
하
면
 
매
우
 
간
단
하
게
 
학
습
과
 
예
측
을
 
할
 
수
 
있
어
서
 
그
냥
 
하
나
의
 
함
수
만
 
만
들
면
 
편
하
게
 
사
용
가
능
하
다
.



```python
d
e
f
 
t
r
a
i
n
_
a
n
d
_
t
e
s
t
(
m
o
d
e
l
)
:

 
 
 
 
m
o
d
e
l
.
f
i
t
(
t
r
a
i
n
_
d
a
t
a
,
 
t
r
a
i
n
_
l
a
b
e
l
)

 
 
 
 
p
r
e
d
i
c
t
i
o
n
 
=
 
m
o
d
e
l
.
p
r
e
d
i
c
t
(
t
e
s
t
_
d
a
t
a
)

 
 
 
 
a
c
c
u
r
a
c
y
 
=
 
r
o
u
n
d
(
m
o
d
e
l
.
s
c
o
r
e
(
t
r
a
i
n
_
d
a
t
a
,
 
t
r
a
i
n
_
l
a
b
e
l
)
 
*
 
1
0
0
,
 
2
)

 
 
 
 
p
r
i
n
t
(
"
A
c
c
u
r
a
c
y
 
:
 
"
,
 
a
c
c
u
r
a
c
y
,
 
"
%
"
)

 
 
 
 
r
e
t
u
r
n
 
p
r
e
d
i
c
t
i
o
n
```

이
 
함
수
에
 
다
섯
가
지
 
모
델
을
 
넣
어
주
면
 
학
습
과
 
평
가
가
 
완
료
된
다
.



```python
#
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n

l
o
g
_
p
r
e
d
 
=
 
t
r
a
i
n
_
a
n
d
_
t
e
s
t
(
L
o
g
i
s
t
i
c
R
e
g
r
e
s
s
i
o
n
(
)
)

#
 
S
V
M

s
v
m
_
p
r
e
d
 
=
 
t
r
a
i
n
_
a
n
d
_
t
e
s
t
(
S
V
C
(
)
)

#
k
N
N

k
n
n
_
p
r
e
d
_
4
 
=
 
t
r
a
i
n
_
a
n
d
_
t
e
s
t
(
K
N
e
i
g
h
b
o
r
s
C
l
a
s
s
i
f
i
e
r
(
n
_
n
e
i
g
h
b
o
r
s
 
=
 
4
)
)

#
 
R
a
n
d
o
m
 
F
o
r
e
s
t

r
f
_
p
r
e
d
 
=
 
t
r
a
i
n
_
a
n
d
_
t
e
s
t
(
R
a
n
d
o
m
F
o
r
e
s
t
C
l
a
s
s
i
f
i
e
r
(
n
_
e
s
t
i
m
a
t
o
r
s
=
1
0
0
)
)

#
 
N
a
v
i
e
 
B
a
y
e
s

n
b
_
p
r
e
d
 
=
 
t
r
a
i
n
_
a
n
d
_
t
e
s
t
(
G
a
u
s
s
i
a
n
N
B
(
)
)
```

#
 
6
.
 
마
무
리


위
에
서
 
볼
 
수
 
있
듯
 
4
번
째
 
모
델
인
 
R
a
n
d
o
m
 
F
o
r
e
s
t
에
서
 
가
장
 
높
은
 
정
확
도
(
8
8
.
5
5
%
)
를
 
보
였
는
데
,
 
이
 
모
델
을
 
채
택
해
서
 
s
u
b
m
i
s
s
i
o
n
 
해
보
자
.



```python
s
u
b
m
i
s
s
i
o
n
 
=
 
p
d
.
D
a
t
a
F
r
a
m
e
(
{

 
 
 
 
"
P
a
s
s
e
n
g
e
r
I
d
"
:
 
t
e
s
t
[
"
P
a
s
s
e
n
g
e
r
I
d
"
]
,

 
 
 
 
"
S
u
r
v
i
v
e
d
"
:
 
r
f
_
p
r
e
d

}
)


s
u
b
m
i
s
s
i
o
n
.
t
o
_
c
s
v
(
'
s
u
b
m
i
s
s
i
o
n
_
r
f
.
c
s
v
'
,
 
i
n
d
e
x
=
F
a
l
s
e
)
```

출
처
:
 
h
t
t
p
s
:
/
/
c
y
c
1
a
m
3
n
.
g
i
t
h
u
b
.
i
o
/
2
0
1
8
/
1
0
/
0
9
/
m
y
-
f
i
r
s
t
-
k
a
g
g
l
e
-
c
o
m
p
e
t
i
t
i
o
n
_
t
i
t
a
n
i
c
.
h
t
m
l
 
를
 
참
고
했
음
.

