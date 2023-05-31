import os 
import io

import ssl
import certifi
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request

import boto3

from gensim.models import Word2Vec
from gensim.models import KeyedVectors


from konlpy.tag import Okt
# from konlpy.tag import Mecab

import pymysql
from dotenv import load_dotenv

from tqdm import tqdm

from datetime import datetime

from itertools import chain
from collections import Counter

# s3에서 모델 기존에 학습된 모델을 호출. 

# env 파일 업로드
load_dotenv()

access_key = os.getenv("AWS_KEY")
secret_key = os.getenv("AWS_SECRET_KEY")

# s3 객체 정의.
# s3 = boto3.resource('s3',
#                   aws_access_key_id=access_key,
#                   aws_secret_access_key=secret_key)

# bucket = s3.Bucket('hangle-square')
# model_file = 'recsys_model/review_w2v'
# local_model_path = 'review_w2v_test'
# bucket.download_file(model_file, local_model_path)
# # 모델 로드
# try:
#     model = KeyedVectors.load_word2vec_format(local_model_path)
#     # model = Word2Vec.load("review_w2v")
# except KeyError:
#     print("모델 로드 X ")

model = Word2Vec.load('test_w2v')

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

# 현재 시간대를 호출 
now = datetime.now()

# 년,월,일을 추출 
print(now.strftime('%Y-%m-%d'))

# RDS에서 추가로 수집된 데이터만 추출 -> 실 서버에서 4일단위로 해야 현재 스펙에 맞게 돌아감 
train_data = pd.read_sql_query(f"select * from News where date IN ('2023-05-25','2023-05-26','2023-05-27','2023-05-28')",conn)
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
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','것','도','를','으로','자','에','와','한','하다','불펌','금지','기사','기자','속보']
# okt 형태소 분석기를 사용한 토큰화 작업

# okt 형태소 모듈 
okt = Okt()

# 토큰화된 데이터를 저장하기 위한 리스트.
tokenized_data = []

for sentence in tqdm(train_data['document']):
    # tokenized_sentence = okt.morphs(sentence) # morphs 메서드를 사용해셔 형태소만을 분리시켜서 반환 -> pos에 비해 속도가 빠름 
    # stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 위에서 명시한 불용어 제거
    tokenized_sentence = okt.pos(sentence)
    stopwords_removed_sentence = [word for word, pos in tokenized_sentence if pos in ['Noun', 'Verb', 'Adjective'] and word not in stopwords]  
    tokenized_data.append(stopwords_removed_sentence)
    
plt.hist([len(review) for review in tokenized_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

# 기존 모델의 단어 집합을 유지하면서 새로운 단어를 추가
model.build_vocab(tokenized_data, update=True)

# train 메서드를 사용하여 새로운 데이터를 추가로 학습
model.train(tokenized_data, total_examples=model.corpus_count, epochs=model.epochs)

# 재학습된 모델 저장
model.save('model/word2vec_updated')
    