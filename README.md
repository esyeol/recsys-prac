### news recsys practice 
<hr>

> setup  

- [colab](https://colab.research.google.com/)
- python 3.10
- 학습에 사용되는 모듈은 colab에 default로 제공되는 모듈 사용. konlpy 모듈만 별도로 설치. (최상단 명시 되어)

> how to

- w2v-model(word2vec)
    - 크롤러로 수집한 데이터 1차 전처리후 csv 형식으로 추출.
    - word2vec 학습을 위한 결측값, 불용어 제거 토큰화 - 
    - 학습(skip-gram, CBOW)
    - 임베딩한 모델 생성후 export 
    - 모델 import 
    - 학습에 사용된 csv 호출 
    - 추천 받고 싶은 idx 문서 입력
    - 입력한 문서 기반 학습된 모델에서 cosine similarity를 통해 상위 5개 추출 
    - 유사도, 제목, 본문, 카테고리 추출 
    <br>
    <br>
- d2v-model(doc2vec)
    - w2v에서 학습한 방법론으로 접근 
    - doc2vec은 문서 단위로 임베딩하기 때문에, 별도의 토큰화 작업만 제외. 
    - 학습 (DM, DBOW)
    - 모델 export 
    - 모델 import 
    - 추천 받는 프로세스는 w2v과 동일 


> 실행 방식 & 프로세스 

1. 프로젝트 fork or clone  

* fork 후 colab 연동 
* local clone 후 google cloud 업로드 후 실행. 

<br>

2. lambda에서 수집중인 데이터 파싱  
* csv 형식으로 추출
* 학습에 필요한 양식으로 전처리 후 colab 폴더에 적재, 경로는 코드 참고
* 파싱 & 1차 전처리는 Proprocessing 내부 코드 참고

<br>

3. 학습된 모델 (w2v, d2v)추출 후 S3로 서빙 

<br>

<참고>
<br>
w2v, d2v 파일의 학습 결과물에 맨 하단 실행 결과를 사용.