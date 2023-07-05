import os 
import io

import ssl
import certifi
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request

from sqlalchemy import create_engine
import boto3

from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from gensim.models import FastText

import pymysql
from dotenv import load_dotenv

from tqdm import tqdm
import pickle

import pandas as pd
from konlpy.tag import Mecab
from rank_bm25 import BM25Okapi

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

access_key = os.getenv("AWS_KEY")
secret_key = os.getenv("AWS_SECRET_KEY")

# s3 객체 정의.
s3 = boto3.resource('s3',
                  aws_access_key_id=access_key,
                  aws_secret_access_key=secret_key)

bucket = s3.Bucket('hangle-square')
model_file = 'recsys_model/bm25_model.pickle'
d2v_file_path = 'recsys_model/review_d2v.model'
local_model_path = 'bm25_model_load.pickle'
local_d2v_path = 'review_d2v.model'
bucket.download_file(model_file, local_model_path)
bucket.download_file(d2v_file_path,local_d2v_path)

# # 모델 로드
# try:
#     loaded_model = KeyedVectors.load_word2vec_format(local_model_path)
# except KeyError:
#     print("모델 로드 X ")

# SQLAlchemy 연결 객체 생성
engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}?charset=utf8mb4', echo=False)

# SQL 쿼리 실행하여 데이터프레임으로 변환
news_data = pd.read_sql_query("select * from News", engine)

# SQL 쿼리 실행하여 데이터프레임으로 변환
news_data = pd.read_sql_query("select * from News", engine)

conn.close()

# 데이터 전처리
news_data['document'] = news_data['title'] + ' ' + news_data['contents']
news_data['label'] = news_data['keyword'].map({'연예': 0, '스포츠': 1, '정치': 2, '국제': 3, '사회': 4, '문화': 5})
news_data = news_data.dropna(how='any')
print(len(news_data))

# BM25 모델 객체 로드
with open('bm25_model_load.pickle', 'rb') as f:
    bm25 = pickle.load(f)

mecab = Mecab()
def recommend_documents(keyword, n=300):
    # 입력된 키워드를 형태소 분석하여 검색어 벡터 생성
    keyword_tokens = mecab.morphs(keyword)
    
    # 문서 검색
    doc_scores = bm25.get_scores(keyword_tokens)
    
    # 상위 n개 문서 추출하여 반환
    top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:n]
    recommended_documents = pd.DataFrame(news_data.iloc[top_n_indices])
    print("추천 항목 리스트")
    print(recommended_documents['title'])
    values = recommended_documents['idx'].tolist()
    print(values)
    return recommended_documents

recommend_documents('일본 방사능')
