from sklearn.datasets import fetch_20newsgroups
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import PCA
from matplotlib import pyplot


"""
Word2Vec, gloVe의 성능을 비교하는 모델
각각의 모델에 SVM을 교체해보면서 비교
SVM계열의 알고리즘은 다음 세가지를 선택 SVC, LinearSVC, SGD
f1 성능 목표 : 95%이상
"""

#None은 all category를 선택
"""
TEST Categories categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
"""
categories = None

#본문에서 제거할 part ('headers', 'footers', 'quotes')
remove = ('headers', 'footers', 'quotes')

#fetch data for training and testing
print(("#"*80)+'\n')
print("Loading 20 newsgroups dataset for categories : ", end='')
print(categories if categories else "all")
print("Preparing for loading 20newgroups")
data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)
data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)
print("**All dataset is loaded\n")
print("Creating list for vector. it takes a few times")

#vector를 생성할 corpus를 생성, 추출할 내용은 subject와 content
#print("\n".join(data_train.data[0].split("\n")[:11]))
t = RegexpTokenizer("[\w]+")
content=[]
for doc in data_train.data:
    content.append(t.tokenize(doc))
print("Created list for vector")
model=Word2Vec(content, size=100, window = 2, min_count=50, workers=4, iter=100, sg=0)
model.init_sims(replace=True)
#(tokenized_contents, size=100, window = 2, min_count=50, workers=4, iter=100, sg=1)
"""
포스태깅된 컨텐츠를 100차원의 벡터로 바꿔라. 주변 단어(window)는 앞뒤로 두개까지 보되, 
코퍼스 내 출현 빈도가 50번 미만인 단어는 분석에서 제외해라. CPU는 쿼드코어를 쓰고 100번 반복 학습해라. 
분석방법론은 CBOW와 Skip-Gram 중 후자를 선택해라.
"""
print("Created vector")
print("Saving model")
print(model)
words=list(model.wv.vocab)
print(words)
#바이너리 형태로 모델 생성 embedding.wv.save_word2vec_format('my.embedding', binary=False)
model.save('com_perform_model_cbow.bin')

#Word2Vec 시각화
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()