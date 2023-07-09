from gensim.models import KeyedVectors
import numpy as np

# Fast Text 모델 로드 
model = KeyedVectors.load_word2vec_format('FastTextModel')

def get_sentence_vector(sentence):
    # 문장을 단어로 분할
    words = sentence.split()
    # 단어의 임베딩 벡터를 가져옴.
    word_vectors = [model.get_vector(word) for word in words if word in model.key_to_index]
    # 문장 벡터를 생성
    sentence_vector = np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)
    
    return sentence_vector

# sentence 값을 기반으로 10개의 유사한 단어를 추출, 하고 유사도 값이 0.1 이하일 경우 제외 
def recommend_similar_words(sentence, topn=10, min_similarity=0.1):
   # 문장을 단어로 분할
    words = sentence.split()
    # 단어의 임베딩 벡터를 가져옴.
    word_vectors = [model.get_vector(word) for word in words if word in model.key_to_index]
    # 문장 벡터를 생성
    sentence_vector = np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)
    
    # 문장 벡터와 유사한 단어 목록 추출 
    similar_words = model.similar_by_vector(sentence_vector, topn=topn)
    
    # 유사도가 min_similarity 이상인 단어만 추출.
    similar_words_filtered = [(word, similarity) for word, similarity in similar_words if similarity >= min_similarity]
    
    # 상위 topn개의 유사한 단어를 반환.
    return similar_words_filtered[:topn]

# input과 유사한 단어를 추출 
input_sentence = '김민재 뮌헨 이적'
similar_words = recommend_similar_words(input_sentence)

# 결과 출력
print(f"입력 문장: {input_sentence}")
print(similar_words)
for word, similarity in similar_words:
    print(f"유사한 단어: {word}, 유사도: {similarity}")