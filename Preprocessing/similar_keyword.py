from gensim.models import KeyedVectors

# 모델 로드
try:
    loaded_model = KeyedVectors.load_word2vec_format("review_w2v")
    model_result = loaded_model.most_similar("keyword")
    for res in model_result:
        word, score = res
        print(f"({word}, {score})\n")
except KeyError as e:
    print(f"KeyError => {e}")