import os 
import io

import ssl
import certifi
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request

import boto3

from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec

from konlpy.tag import Okt

import pymysql
from dotenv import load_dotenv

from tqdm import tqdm

# s3 객체 정의.
s3_client = boto3.client('s3')

# env 파일 업로드
load_dotenv()
# 환경 변수에서 DB 연결 정보 로드
DB_HOST = os.getenv("N_DB_HOST")
DB_USER = os.getenv("N_DB_USER")
DB_PASSWORD = os.getenv("N_DB_PWD")
DB_NAME = os.getenv("N_DB_NAME")
DB_PORT = os.getenv("DB_PORT")

conn = pymysql.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    db=DB_NAME,
    charset='utf8'
)
cur = conn.cursor()

# 크롤러에서 수집되는 데이터량 증가로, 모델학습에 사용되지 않은 날짜값을 기반으로 데이터 조회후, online 학습을 위한 새로운 모델 생성후 재학습
train_data = pd.read_sql_query("SELECT * FROM News WHERE date NOT IN ('2023-05-24','2023-05-25','2023-05-26','2023-05-27','2023-05-28','2023-05-29','2023-05-30') ", conn)
print(train_data)

# "title" 열에서 필요없는 문자 제거
train_data["title"] = train_data["title"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)

# "content" 열에서 필요없는 문자 제거
train_data["contents"] = train_data["contents"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)

# idx 행의 값을 재배열  ex) idx 2XXXX -> 1
train_data["idx"] = range(1, len(train_data)+1)

# title, contents 값을 document라는 새로운 컬럼으로 생성
train_data['document'] = train_data['title']+' '+train_data['contents']

#keyword 라는 컬럼을 숫자로 매핑하여 labeling 작업 => 별도로 label이라는 컬럼을 생성하여 적용
train_data['label'] = train_data['keyword'].map({'연예': 0, '스포츠': 1, '정치': 2, '국제': 3, '사회': 4, '문화': 5})

print(train_data)

conn.close()

# title, contents 값을 document라는 새로운 컬럼으로 생성
train_data['document'] = train_data['title']+' '+train_data['contents']

#keyword 라는 컬럼을 숫자로 매핑하여 labeling 작업 => 별도로 label이라는 컬럼을 생성하여 적용
train_data['label'] = train_data['keyword'].map({'연예': 0, '스포츠': 1, '정치': 2, '국제': 3, '사회': 4, '문화': 5})

# train data 에서 상위 5개 출력
print(train_data[:5])
# 총 리뷰 갯수 확인
print(len(train_data))
# 결측값 확인
print(train_data.isnull().values.any())
# 결측값이 존재하기에 결측값이 존재하는 행을 모두 제거 => NULL 값이 존재하는 Row 모두제거
train_data = train_data.dropna(how='any')
print(len(train_data))

# word2Vec 학습시, 학습에 사용하고 싶지 않은 단어들인 불용어 제거 
stopwords = ['의','가','것','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','불펌','금지','기사','기자','속보']
# okt 형태소 분석기를 사용한 토큰화 작업

# okt to mecab 
okt = Okt()

# 토큰화된 데이터를 저장하기 위한 리스트.
tokenized_data = []

for sentence in tqdm(train_data['document']):
    tokenized_sentence = okt.morphs(sentence) # morphs || pos 활용은 고려 현재는 속도 때문에 morphs 메서드 사용 
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 위에서 명시한 불용어 제거
    tokenized_data.append(stopwords_removed_sentence)
    

print('리뷰의 최대 길이 :',max(len(review) for review in tokenized_data))
print('리뷰의 평균 길이 :',sum(map(len, tokenized_data))/len(tokenized_data))

plt.hist([len(review) for review in tokenized_data], bins=50)
plt.show()
'''
 토큰화된 뉴스 데이터를 학습
 size => 만들어질 워드 벡터의 차원 => 100 * 100
 window => 컨텍스트 윈도우의 크기, 컨텍스트 윈도우는 단어 앞과 뒤에서 몇개 단어를 볼것인지를 정하는 크기임. 
 min_count => 단어 최소 빈도수의 임계치 ( 해당 임계치 보다 적은 단어는 훈련시키지 않음 ) 
 workker => 학습에 사용하는 프로세스 갯수 
 sg => 0 일 경우, CBOW, 1일 경우 Skip-Gram => word2vec 알고리즘의 형식
'''
model = Word2Vec(sentences=tqdm(tokenized_data, desc="Training"), vector_size=100, window=5, min_count=5, workers=6, sg=1)

model.wv.vectors.shape
model.save('model/test_w2v')

# # 모델을 review_w2v 이라는 모델명으로 저장
# model.wv.save_word2vec_format('review_w2v')

# S3 업로드
# with open('review_w2v', 'rb') as data:
#     s3_client.put_object(Bucket='hangle-square', Key='recsys_model/review_w2v', Body=data)

# model.wv.save_word2vec_format('model/review_w2v')
# s3_client.upload_file('model/review_w2v', 'hangle-square', 'recsys_model/review_w2v')


print(model.wv.vectors.shape)
