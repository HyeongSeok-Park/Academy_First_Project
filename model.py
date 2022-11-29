import numpy as np
import pandas as pd
import warnings
from gensim.models.word2vec import Word2Vec
# from konlpy.tag import Okt
from konlpy.tag import Kkma
from gensim.models import Word2Vec
from tqdm import tqdm

## 유사 법률용어 검색 모델 만들기 ##

law = pd.read_csv('법무부_(생활법률지식)1.법률용어_20191231.csv', encoding='euc-kr')

law['설명'] = law['설명'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    # 정규 표현식을 통한 한글 외 문자 제거

# okt = Okt() # 연산속도는 빠르나, 정확도가 떨어진다.
kkma = Kkma() # 정확도는 높으나, 연산속도가 느리다.

def word_edit(a_list) :
    n_words = ['은','는','이','가','게','의','들','과','도','로','를','을','으로','에','와','한','하다','하는','것','이다','있다','수','란','이란','그','후','하나','요','너무','심해','있게','입니다','합니다','됩니다','등']
        # 불용어 제거
    
    value = []

    for explain in tqdm(a_list, desc='진행상황',ncols=100, leave=True) :
        # pos = okt.morphs(explain)
        pos = kkma.nouns(explain)
        removed_words = [word for word in pos if not word in n_words]
        value.append(removed_words)

    return value

result_pos = word_edit(law['설명'])
print(result_pos)


## 법무부_(생활법률지식)을 학습한 모델 만들기 ##

model = Word2Vec(sentences=result_pos, vector_size=100, window=10, min_count=1, workers=2, sg=1)
    # sentences = 리스트 형태의 데이터
    # vector_size = 임베딩 된 벡터의 차원 (워드 벡터의 특징 값)
    # window = 학습할 주변 단어의 수
    # min_count = 사용할 단어의 최소 빈도 (빈도 이하의 단어 무시)
    # workers = 동시에 처리할 작업 수 (코어 수와 비슷하게 설정)
    # sg = 0은 CBOW, 1은 Skip-gram

model.save('word2vec.model')
    # 학습 모델 파일 저장

model = Word2Vec.load('word2vec.model')
    # 학습 모델 파일 불러오기

print(type(model))
print(model.wv.vectors.shape)


## 사용자 발화문 입력 ##

user_barhwa = ['층간소음이 너무 심해요']
okt_user_barhwa = word_edit(user_barhwa)


## 법무부_(생활법률지식)에서 가장 유사한 법률용어 찾기 ##

add_list = []

for i in range(len(law)) :
    law_list = [law['설명'][i]]
    data_similar = model.wv.n_similarity(okt_user_barhwa[0], word_edit(law_list)[0])
    add_list.append(data_similar)

law['유사도'] = add_list
new_law = law.sort_values('유사도',ascending=False)[:3]
new_law

######################################################################################################
print('사용자께서 입력하신' + ' "'+ user_barhwa[0]+'" ' + '와 가장 유사한 법률 용어는 아래와 같습니다.')
print('법률용어 :' + new_law.iloc[0,1])
print('설명 :' + new_law.iloc[0,2])




# s2 = result_pos



# 법률 용어 데이터
# model_law = Word2Vec.load("/workspace/ChatBot_test/law_word2vec.model")
# law = pd.read_csv("/workspace/ChatBot_test/law.csv", encoding="euc-kr")
# result_lawfile = open('/workspace/ChatBot_test/result.txt', 'r')
# result_pos_law = result_lawfile.read()
# result_law = ast.literal_eval(result_pos_law)
# # 뉴스 기사 데이터
# model_news = Word2Vec.load("/workspace/ChatBot_test/model_news.model")
# news = pd.read_csv("/workspace/ChatBot_test/news_half.csv", encoding="utf-8")
# result_newsfile = open('/workspace/ChatBot_test/news_pos_half.txt', 'r')
# result_pos_news = result_newsfile.read()
# result_news = ast.literal_eval(result_pos_news)

# @application.route('/test11', methods=['POST'])
# def Checklaw1():    
#     content = request.get_json()
#     sys_test11 = content["userRequest"]["utterance"]
#     # sys_test8 = content["action"]["params"]["sys_test8"]
    
#     #법률 용어 형태소 분석
#     s1 = kkma.nouns(sys_test11)
#     s2 = result_law
#     #뉴스기사 용어
#     s3 = result_news
    
#     answer_law=[]
#     answer_news=[]
    
#     #법률용어 유사도 측정
#     for i in s2:
#         distance_law = model_law.wv.n_similarity(s1, i)
#         answer_law.append([distance_law])
#     #뉴스기사 유사도 측정
#     # try:
#     for j in s3:
#         distance_news = model_news.wv.n_similarity(s1, j)
#         answer_news.append([distance_news])
#     # except KeyError:
#     #     print(str(sys_test8)+"과 관련된 기사가 없습니다.")

#     #가장 유사한 법률용어 찾기
#     ds_law = {'Explain': (answer_law)}
#     df_law = pd.DataFrame(ds_law)
#     law_max = df_law.sort_values("Explain",ascending=False)[:1].index.values[0]
#     law_word = law.loc[law_max,'word']
#     law_explain = law.loc[law_max,'explain']
#     #가장 유사한 뉴스기사 찾기
#     ds_news = {'Title': (answer_news)}
#     df_news = pd.DataFrame(ds_news)
#     news_max = df_news.sort_values("Title",ascending=False)[:1].index.values[0]
#     news_title = news.loc[news_max,'title']
#     news_url = news.loc[news_max,'url']
    
#     answer_1 = str(law_word)+" : "+str(law_explain)
#     answer_2 = str(news_title)+" : "+str(news_url)
#     # answer_2 = "https://www.naver.com"
#     print(sys_test11)
#     print(answer_1)
#     print(answer_2)
    