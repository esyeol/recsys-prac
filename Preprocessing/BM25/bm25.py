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
from gensim.models import FastText

from konlpy.tag import Okt
# from konlpy.tag import Mecab

import pymysql
from dotenv import load_dotenv

from tqdm import tqdm

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

# RDS에서 데이터 호출 
train_data = pd.read_sql_query("select * from News",conn)

print(f'가져온 데이터 길이{train_data}')

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
train_data = train_data.dropna(how='any')
print(train_data)

conn.close()

# 토큰화된 데이터 생성
mecab = Mecab()
tokenized_data = [mecab.morphs(sentence) for sentence in train_data['document']]

# BM25 모델 학습
bm25 = BM25Okapi(tokenized_data)

def recommend_documents(keyword, n=300):
    # 입력된 키워드를 형태소 분석하여 검색어 벡터 생성
    keyword_tokens = mecab.morphs(keyword)
    
    # 문서 검색
    doc_scores = bm25.get_scores(keyword_tokens)
    
    # 상위 n개 문서 추출하여 반환
    top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:n]
    recommended_documents = pd.DataFrame(train_data.iloc[top_n_indices])
    print("추천 항목 리스트")
    print(recommended_documents['title'])
    values = recommended_documents['idx'].tolist()
    print(values)
    return recommended_documents

recommend_documents('스포츠')
