from konlpy.tag import Mecab

mecab = Mecab()

'''
자모음이 포함되어 있는 경우 저장하지 않도록 처리하는 로직 
'''
def has_consonant_or_vowel(word):
    # 여기서는 한글 음절의 자음과 모음을 확인.
    consonants = set("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
    vowels = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")

    for char in word:
        if char in consonants or char in vowels:
            return False
    return True

'''
명사 또는 대명사인지를 판별하는 로직
'''
def is_noun_or_pronoun(word):
    # 여기서는 명사 또는 대명사로 판단되는 품사 태그를 사용
    pos = mecab.pos(word)
    for _, tag in pos:
        if tag.startswith("N") or tag.startswith("P"):
            return True
    return False


'''
최종적으로 디비에 저장하는 로직 
유저가 입력한 키워드가 명사,대명사이고 단자모음이 포함 되어있지 않고, 길이가 1이상일 때만 최종적으로 DB에 저장하는 로직 
'''
def save_to_db(data):
    if is_noun_or_pronoun(data) and len(data) > 1 and has_consonant_or_vowel(data):
        print("명사나 대명사이기 때문에 데이터를 DB에 저장합니다:", data)
    else:
        print('명사나 대명사가 아님 ')

# 사용자 입력 데이터
user_input = "테스트"

# 데이터가 명사 또는 대명사인지를 판별하고, 해당 경우에만 DB에 저장
save_to_db(user_input)
