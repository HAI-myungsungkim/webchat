from gensim.models import doc2vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
#형태소 분석
import jpype
from konlpy.tag import Kkma

from gensim.models import word2vec;print("FAST_VERSION", word2vec.FAST_VERSION)

# # 파일로부터 모델을 읽는다. 없으면 생성한다.
# try:
#     d2v_faqs = Doc2Vec.load('d2v_faqs.model')
#     faqs = pd.read_csv('web_data1.csv')
# except:
#     faqs = pd.read_csv('web_data1.csv')

kkma = Kkma()
filter_kkma = ['NNG',  #보통명사
             'NNP',  #고유명사
             'OL' ,  #외국어
            ]

def tokenize_kkma(doc):
    jpype.attachThreadToJVM()
    token_doc = ['/'.join(word) for word in kkma.pos(doc)]
    return token_doc

def tokenize_kkma_noun(doc):
    jpype.attachThreadToJVM()
    token_doc = ['/'.join(word) for word in kkma.pos(doc) if word[1] in filter_kkma]
    return token_doc

# 파일로부터 모델을 읽는다. 없으면 생성한다.
try:
    d2v_faqs = Doc2Vec.load('d2v_faqs_al1.model')
    faqs = pd.read_csv('web_data1.csv', encoding='CP949')
except:
    faqs = pd.read_csv('web_data1.csv', encoding='CP949')
    # 토근화

    # 리스트에서 각 문장부분 토큰화
    token_faqs = []
    for i in range(len(faqs)):
        token_faqs.append([tokenize_kkma_noun(faqs['질문'][i]), i])

    # Doc2Vec에서 사용하는 태그문서형으로 변경
    tagged_faqs = [TaggedDocument(d, [int(c)]) for d, c in token_faqs]

    # make model
    import multiprocessing
    cores = multiprocessing.cpu_count()
    d2v_faqs = doc2vec.Doc2Vec(
                                    vector_size=100,
                                    alpha=0.025,
                                    min_alpha=0.025,
                                    hs=1,
                                    negative=0,
                                    dm=0,
                                    window=3,
                                    dbow_words=1,
                                    min_count=1,
                                    workers=cores,
                                    seed=0,
                                    epochs=100
                                    )
    d2v_faqs.build_vocab(tagged_faqs)
    d2v_faqs.train(tagged_faqs,
                   total_examples=d2v_faqs.corpus_count,
                   epochs=d2v_faqs.epochs)

    d2v_faqs.save('d2v_faqs_al1.model')

# 챗봇 답변
try:
    d2v_wellness = Doc2Vec.load('wellness_res.model')
    wellness_chat = pd.read_csv('wellness_chat.csv', encoding='CP949')
except:
    wellness_chat = pd.read_csv('wellness_chat.csv', encoding='CP949')
    # 토근화

    # 리스트에서 각 문장부분 토큰화
    token_faqs = []
    for i in range(len(faqs)):
        token_faqs.append([tokenize_kkma_noun(wellness_chat['유저'][i]), i])

    # Doc2Vec에서 사용하는 태그문서형으로 변경
    tagged_faqs = [TaggedDocument(d, [int(c)]) for d, c in token_faqs]

    # make model
    import multiprocessing
    cores = multiprocessing.cpu_count()
    d2v_wellness = doc2vec.Doc2Vec(
                                    vector_size=100,
                                    alpha=0.025,
                                    min_alpha=0.025,
                                    hs=1,
                                    negative=0,
                                    dm=0,
                                    window=3,
                                    dbow_words=1,
                                    min_count=1,
                                    workers=cores,
                                    seed=0,
                                    epochs=100
                                    )
    d2v_wellness.build_vocab(tagged_faqs)
    d2v_wellness.train(tagged_faqs,
                   total_examples=d2v_wellness.corpus_count,
                   epochs=d2v_wellness.epochs)

    d2v_wellness.save('wellness_res.model')


# FAQ 답변
def faq_answer(input):
    # 테스트하는 문장도 같은 전처리를 해준다.
    tokened_test_string = tokenize_kkma_noun(input)
    print('인풋!!' + str(tokened_test_string))
    topn = 10
    test_vector = d2v_faqs.infer_vector(tokened_test_string)
    result = d2v_faqs.docvecs.most_similar([test_vector], topn=topn)
    answer_list = []

    for i in range(topn):
        print("{}위. {}, {} {} {}".format(i + 1, result[i][1], result[i][0], faqs['질문'][result[i][0]], faqs['답변'][result[i][0]]))
        answer_list.append(dict(acc=result[i][1], question=faqs['질문'][result[i][0]], answer=faqs['답변'][result[i][0]]))
    #
    test_vector_for_res = d2v_wellness.infer_vector(tokened_test_string)
    result_for_res = d2v_wellness.docvecs.most_similar([test_vector_for_res], topn=1)
    answer_list.append(dict(acc=result_for_res[0][1], question=wellness_chat['유저'][result_for_res[0][0]], answer=wellness_chat['챗봇'][result_for_res[0][0]]))

    return dict(acc1=result[0][1], question1=faqs['질문'][result[0][0]],answer1=faqs['답변'][result[0][0]],
                acc2=result[1][1], question2=faqs['질문'][result[1][0]],answer2=faqs['답변'][result[1][0]],
                acc3=result[2][1], question3=faqs['질문'][result[2][0]],answer3=faqs['답변'][result[2][0]],
                acc4=result[3][1], question4=faqs['질문'][result[3][0]],answer4=faqs['답변'][result[3][0]],
                acc5=result[4][1], question5=faqs['질문'][result[4][0]],answer5=faqs['답변'][result[4][0]],
                acc6=result[5][1], question6=faqs['질문'][result[5][0]],answer6=faqs['답변'][result[5][0]],
                acc7=result[6][1], question7=faqs['질문'][result[6][0]],answer7=faqs['답변'][result[6][0]],
                acc8=result[7][1], question8=faqs['질문'][result[7][0]],answer8=faqs['답변'][result[7][0]],
                acc9=result[8][1], question9=faqs['질문'][result[8][0]],answer9=faqs['답변'][result[8][0]],
                acc10=result[9][1], question10=faqs['질문'][result[9][0]],answer10=faqs['답변'][result[9][0]],
                acc11=result_for_res[0][1], question11=wellness_chat['유저'][result_for_res[0][0]],answer11=wellness_chat['챗봇'][result_for_res[0][0]])


def faq_search(inputs):
    keywords = None
    for word in inputs:
        if keywords is None:
            keywords = word
        else:
            keywords = keywords + '|' + word
    faqs[faqs.str.contains(keywords)]
    print(faqs)
    return 0