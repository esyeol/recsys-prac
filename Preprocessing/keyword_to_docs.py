import pandas as pd
from konlpy.tag import Mecab
from gensim.models.doc2vec import Doc2Vec
import numpy as np

'''
keyword => docs 추천을 위한 입력 값
n => 추천 받을 문서  
'''
def recommend_documents(keyword, n):
    mecab = Mecab()
    model = Doc2Vec.load("review_d2v.model")

    # 입력된 키워드를 형태소 분석하여 doc2vec 모델에서 유사한 벡터 추출 (mecab 형태소 분석 모델 사용)
    keyword_vector = model.infer_vector(mecab.morphs(keyword))

    # 모든 문서와의 유사도 계산
    doc_vectors = [model.docvecs[str(i)] for i in range(len(model.docvecs))]
    # numpy 모듈의 np.dot() 메서드를 사용하여 학습된 모든 문서 벡터 + 입력된 키워드간 내적 계산
    similarities = np.dot(doc_vectors, keyword_vector)

    # 상위 n(5)개 문서 추출하여 반환
    top_n_indices = np.argsort(similarities)[::-1][:n]
    recommended_documents = pd.DataFrame(train_data.iloc[top_n_indices])
    return recommended_documents


# 특정 키워드 입력시 입력한 키워드와 학습된 모든 문서 벡터를 기반으로 유사한 문서 추천 테스트
recommend_documents('키워드',20)