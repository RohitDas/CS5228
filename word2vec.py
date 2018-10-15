from bow import read_news_articles, read_titles, complete_preprocessing, augment_with_title

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import numpy as np

def tag_docs(tokens_list):
    tagged_docs = []
    for index, tokens_sents in tokens_list:
        token_list = []
        for tokens_sent in tokens_sents: 
            token_list += tokens_sent
        tagged_docs.append(TaggedDocument(words=token_list, tags=[str(index)]))
    return tagged_docs


def create_doc2vec(tag_docs):
    """
        Trains the doc2vec model
    """
    max_epochs = 100                                
    vec_size = 500                                            
    alpha = 0.025                          
    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,                                                                               
                    min_alpha=0.025,
                    min_count=0)
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
articles = read_news_articles()
print("Reading titles")
titles = read_titles()
print("Processing articles")
prepreprocessed_articles = complete_preprocessing(articles)
print("processing titles")
preprocessed_titles = complete_preprocessing(titles)
print("augmenting with titles")
full_representation = augment_with_title(prepreprocessed_articles, preprocessed_titles)
tag_docs = tag_docs(full_representation)
print tag_docs[0]
model = create_doc2vec(tag_docs)
print(model.docvecs.vectors_docs)
print(model.docvecs[2000])
np.save("word2vec", model.docvecs.vectors_docs)
print len(tag_docs)


