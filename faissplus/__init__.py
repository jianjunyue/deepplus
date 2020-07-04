#简单的cosine_similarity的计算
import faiss
from faiss import normalize_L2
import numpy as np
def cosine_similar():
    '''
    cosine_similarity
    use

    :return:
    '''
    d = 64                           # dimension
    nb = 105                    # database size
    #主要是为了测试不是归一化的vector
    training_vectors= np.random.random((nb, d)).astype('float32')*10
    print('just  compare with skearn')
    from sklearn.metrics.pairwise import cosine_similarity
    #主要是为了与sklearn 比较结果
    ag=cosine_similarity(training_vectors)
    fe=np.sort(ag,axis=1)
    print('normalize_L2')
    normalize_L2(training_vectors)
    print('IndexFlatIP')
    index=faiss.IndexFlatIP(d)
    index.train(training_vectors)
    print(index)
    print('train')
    print(index.is_trained)
    print('add')
    print(index)
    index.add(training_vectors)
    print('search')
    D, I =index.search(training_vectors[:100], 5)
    print(I[:5])                   # 表示最相近的前5个的index
    print(D[:5])  # 表示最相近的前5个的相似度的值

cosine_similar()