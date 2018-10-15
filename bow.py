"""
    Vocabulary builder from titles column.
"""
import re
import numpy as np
import pickle
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
from nltk.stem import WordNetLemmatizer

def read_titles():
    """
        Function read the titles file and stores it in a list in the format (id, title)
        where id is the docid
    """
    titles = []
    with open("titles", "r") as fp:
        for line in fp:
            try:
                id, sent = line.lower().split("|")
            except:
                splitted = line.lower().split("|")
                id,sent = splitted[0], ' '.join(splitted[1:])
                
            titles.append((id, [sent.strip()]))
    return titles

def read_news_articles():
    """
        Function reads the news articles files : Thread_[0..3]
        the files contain the articles in the following formats: id_1|article_1##id_2|article_2 ....
    """
    num_files = 4
    lines = []
    for num_file in range(num_files):
        file_path = "test_data/Thread_" + str(num_file) + ".dat"
        with open(file_path, "r") as fp:
            lines += fp.read().split("##")
    lines = map(lambda line: line.split("|"), lines)
    lines =  filter(lambda tup: len(tup) == 2, lines)
    #filter all those lines that have Empty as there article text
    lines = filter(lambda tup: tup[1].strip() != 'Empty', lines)
    #SPlit each article into lines.
    lines = map(lambda tup: (tup[0], tup[1].lower().split("\n")), lines)
    return lines

def preprocess(lines):
    """
        Does some preprocessing.
    """
    processed_lines = []
    for line in lines:
        #Remove non-alphanumeric characters
        processed_line = re.sub(r'\W+', ' ', line).strip()
        #Remove all digits
        processed_line = re.sub(r'\w*\d\w*', '', processed_line).strip()
        if processed_line:
            processed_lines.append(processed_line)
    return processed_lines
 
def tokenize(lines):
    """
        Uses nltk word_tokenize to tokenize the lines
    """
    tokenized_lines = []
    for line in lines:
        tokenized_lines.append(word_tokenize(line))
    return tokenized_lines

def remove_stopwords(tokenized_lines):
    """
        Remove all the stopwords
    """
    lines = []
    stop_words = stopwords.words("english")
    for line in tokenized_lines:
        lines.append([word for word in line if word not in stop_words])
    return lines

def stem(tokens_list):
    """
        Uses Porter Stemmmingalgorithm to stem the lines.
    """
    stemmed_list = []
    p_stemmer = PorterStemmer()
    for token_list in tokens_list:
        stemmed_list.append([p_stemmer.stem(i) for i in token_list])
    return stemmed_list

def lemmatize(tokens_list):
    """
        Uses WordNet lemmatizer to lemmatize
    """
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_list = []
    for token_list in tokens_list:
        lemmatized_list.append([wordnet_lemmatizer.lemmatize(i) for i in token_list])
    return lemmatized_list

def tag_docs(tokens_list):
    """
        Creates a list of Tagged documents
    """
    tagged_docs = []
    for index, token_list in enumerate(tokens_list):
        tagged_docs.append(TaggedDocument(token_list, str(index)))
    return tagged_docs

def get_representation(preliminary_count, size):
    """
        It gets the bag-of-words representation of a word.
    """
    representation = []
    for idx, count in preliminary_count:
        if len(representation) == idx:
            representation.append(count)
        else:
            while len(representation) < idx:
                representation.append(0)
            representation.append(count)

    while len(representation) < size:
        representation.append(0)
    return representation

def create_bow(stemmed_tokens):
    """
        Creates the bow representation for tokens
    """
    texts = []
    for id, lines in stemmed_tokens:
        texts += lines
    print texts
    dictionary = corpora.Dictionary(texts)
    dictionary.save('dictionary.dict')
    
    size_of_dictionary = len(dictionary)
    #Get the doc 2 bag-of-words model
    bow_representation = []
    id_list = []
    for id, lines in stemmed_tokens:
        para = []
        for line in lines:
            para += line
        id_list.append(id)
        bow_representation.append(get_representation(dictionary.doc2bow(para), size_of_dictionary))
    return np.array(id_list), np.array(bow_representation)

def complete_preprocessing(lines):
    """
        Does a series of preprocessing steps.
    """
    lines = map(lambda tup: (tup[0], preprocess(tup[1])) if tup[1] != 'Empty' else tup, lines)
    lines = map(lambda tup: (tup[0], tokenize(tup[1])), lines)
    lines = map(lambda tup: (tup[0], remove_stopwords(tup[1])), lines)
    #lines = map(lambda tup: (tup[0], stem(tup[1])), stopword_removed_lines)
    #lines = map(lambda tup: (tup[0], lemmatize(tup[1])), stopword_removed_lines)
    return lines

def augment_with_title(lines, all_lines):
    """
        For all those documents that had bad urls, use the title instead
    """
    out = []
    curr_id = 0
    for id, text in lines:
        if id == str(curr_id):
            out.append((id, text))
        else:
            while str(curr_id) != id:
                out.append(all_lines[curr_id])
                curr_id += 1
            out.append((id, text))
        curr_id += 1
    return out

print("Reading articles")
articles = read_news_articles()
print("Reading titles")
titles = read_titles()
#print("Processing articles")
prepreprocessed_articles = complete_preprocessing(articles)
print("processing titles")
preprocessed_titles = complete_preprocessing(titles)
print("augmenting with titles")
full_representation = augment_with_title(prepreprocessed_articles, preprocessed_titles)
#print(full_representation)
id, bow = create_bow(preprocessed_titles)
print(id.shape, bow.shape)
np.save("ids", id)
np.save("bow_titles", bow)


#print stopword_removed_titles
#model = create_doc_two_vec(tagged_docs)
#model.save("save/trained.model")
#model.save_word2vec_format('save/trained.word2vec')
