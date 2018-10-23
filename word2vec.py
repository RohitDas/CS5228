from bow import complete_preprocessing

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import numpy as np

def read_titles():
    """
        Function read the titles file and stores it in a list in the format (id, title)
        where id is the docid
    """
    titles = {}
    with open("train_v_title", "r") as fp:
        for line in fp:
            id, title = line.lower().split("\t")
            titles.update({
                id: title.strip()
            })
    return titles

def read_news_articles():
    """
        Function reads the news articles files : Thread_[0..3]
        the files contain the articles in the following formats: id_1|article_1##id_2|article_2 ....
    """
    num_files = 6
    titles =  read_titles()
    lines = []
    for num_file in range(num_files):
        file_path = "test_data_v2/Thread_" + str(num_file) + ".dat"
        with open(file_path, "r") as fp:
            lines += fp.read().split("###")
    lines = map(lambda line: line.split("||"), lines)
    articles  = {}
    empty = 0
    for line in lines:
        if len(line) >= 2:
            id, others = line[0], ' '.join(line[1:]).strip()
            if others == "Empty":
                others = titles[id]
                empty += 1
            else:
                others = titles[id] + " " + others
            articles.update({
                id: others.lower().split("\n")
            })
    return articles

def tag_docs(tokens_list):
    tagged_docs = []
    for index, tokens_sents in enumerate(tokens_list):
        token_list = []
        print tokens_sents
        for token_sent in tokens_sents:
            token_list += token_sent
        tagged_docs.append(TaggedDocument(words=token_list, tags=[str(index)]))
    return tagged_docs


def create_doc2vec(tag_docs):
    """
        Trains the doc2vec model
    """
    max_epochs = 300                                
    vec_size = 300                                            
    alpha = 0.025                          
    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,                                                                               
                    min_alpha=0.025,
                    min_count=1)
    model.build_vocab(tag_docs)

    for epoch in range(max_epochs):
             print('iteration {0}'.format(epoch))
             model.train(tag_docs,
                         total_examples=len(tag_docs),
                         epochs=model.iter)
             # decrease the learning rate
             model.alpha -= 0.0002         
             # fix the learning rate, no decay
             model.min_alpha = model.alpha
    return model



print("Reading articles")
articles_map = read_news_articles()
ids, articles = [], []
for id, article in articles_map.iteritems():
    ids.append(id)
    articles.append(article)
print("Processing articles")
prepreprocessed_articles = complete_preprocessing(articles)
print("processing titles")
#preprocessed_titles = complete_preprocessing(titles)
print("augmenting with titles")
#full_representation = augment_with_title(prepreprocessed_articles, preprocessed_titles)
tag_docs = tag_docs(prepreprocessed_articles)
print tag_docs[0]
model = create_doc2vec(tag_docs)
np.save("word2vec", model.docvecs.vectors_docs)
np.save("ids", np.array(ids))
print len(tag_docs)

