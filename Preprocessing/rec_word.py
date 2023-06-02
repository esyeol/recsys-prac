# 유저가 입력한 키워드에 대해서 유사도가 높은 데이터 상위 6개를 추출해서 추천해주는 부분
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
# 모델 로드
try:
    # loaded_model = Word2Vec.load('test_w2v')
    loaded_model = Word2Vec.load('model/word2vec_updated')
except KeyError:
    print("모델 로드 X ")

# 모델 로드
try:
    # 다중 키워드를 기반으로 유사도 높은 단어 리스트 업 
    keywords = ["축구", "손흥민", "김민재"]
    similar_words = []
    for keyword in keywords:
        similar_words += loaded_model.wv.similar_by_word(keyword, topn=10)
        
    # 유사도 수치를 기준으로 내림차순 정렬 후 상위 6개 단어 추출
    top_similar_words = sorted(similar_words, key=lambda x: x[1], reverse=True)[:6]
    
    # 단어 + 유사도 추출 
    for word, similarity in top_similar_words:
        print(f"({word}, {similarity})\n")
    
except KeyError:
    print("입력된 단어에 대한 유사 단어를 찾을 수 없습니다.")