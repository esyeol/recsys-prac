import pandas as pd
import re
# csv 파일 읽어오기
df = pd.read_csv("data/news.csv")

# 학습에 필요없는 "summary", "link", "imgurl", "date" 데이터 삭제
df.drop(columns=["summary", "link", "imgUrl", "date"], inplace=True)

# "title" 열에서 필요없는 문자 제거
df["title"] = df["title"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)

# "content" 열에서 필요없는 문자 제거
df["contents"] = df["contents"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)

# idx 행의 값을 재배열  ex) idx 2XXXX -> 1
df["idx"] = range(1, len(df)+1)

# 수정된 DataFrame을 csv 파일로 저장
df.to_csv("news.csv", index=False)

# 변경한 csv 파일 읽어 들이기
n_df = pd.read_csv("news.csv")

# title, contents 값을 document라는 새로운 컬럼으로 생성
n_df['document'] = n_df['title']+' '+n_df['contents']

#keyword 라는 컬럼을 숫자로 매핑하여 labeling 작업 => 별도로 label이라는 컬럼을 생성하여 적용
n_df['label'] = n_df['keyword'].map({'연애': 0, '스포츠': 1, '정치': 2, '국제': 3, '사회': 4, '문화': 5})

print(n_df)
