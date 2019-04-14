from gensim.models import Word2Vec
import cPickle as pkl
import six.moves.cPickle as pickle
embed_d = 128
win_L = 5
data_path = '../data/AMiner-T-2013'


def load_data(path = data_path + "content.pkl", maxlen = None, n_words = 600000, sort_by_len = False):
    content_file = open(path, 'rb')
    content_set = pickle.load(content_file)
    content_file.close()

    if maxlen:
        new_content_set_x = []
        new_content_set_y = []
        for x, y in zip(content_set[0], content_set[1]):
            if len(x) < maxlen:
                new_content_set_x.append(x)
                new_content_set_y.append(y)
        content_set = (new_content_set_x, new_content_set_y)
        del new_content_set_x, new_content_set_y

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    content_set_x, content_set_y = content_set

    content_set_x = remove_unk(content_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(content_set_x)
        content_set_x = [content_set_x[i] for i in sorted_index]
        content_set_y = [content_set_y[i] for i in sorted_index]

    content = (content_set_x, content_set_y)

    return content


def word2vec_run():
    content = load_data(data_path + '/content.pkl')
    content_data, content_index = content

    corpus =[]
    for i in range(len(content_data)):
       sentence=[]
       for j in range(len(content_data[i])):
           sentence.append(content_data[i][j])
       corpus.append(sentence)

    w2v_model = Word2Vec(corpus, size = embed_d, window = win_L, min_count = 0, workers = 2, sg = 1)
    
    print("Output word embedding...")
    w2v_model.wv.save_word2vec_format(data_path + "/word_embedding.txt")


word2vec_run()





