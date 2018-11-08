import re
import difflib
from collections import Counter
import pandas as pd
import numpy as np
from sklearn import model_selection
from nltk.corpus import wordnet
from nltk import WordNetLemmatizer
from sklearn.metrics import accuracy_score

TOP_N = 3


def get_pos(word):
    w_synsets = wordnet.synsets (word)
    pos_counts = Counter ()
    pos_counts["n"] = len ([item for item in w_synsets if item.pos () == "n"])
    pos_counts["v"] = len ([item for item in w_synsets if item.pos () == "v"])
    pos_counts["a"] = len ([item for item in w_synsets if item.pos () == "a"])
    pos_counts["r"] = len ([item for item in w_synsets if item.pos () == "r"])

    most_common_pos_list = pos_counts.most_common (3)
    return most_common_pos_list[0][
        0]  # first indexer for getting the top POS from list, second indexer for getting POS from tuple( POS: count )


def lemmatize(tokens_list):
    """
        Uses WordNet lemmatizer to lemmatize
    """
    wordnet_lemmatizer = WordNetLemmatizer ()
    lemmatized_list = []
    for i in tokens_list:
        lemmatized_list.append (wordnet_lemmatizer.lemmatize (i, get_pos (i)))
    return lemmatized_list


def preprocess(text):
    # Remove non-alphanumeric characters
    processed_line = re.sub (r'\W+', ' ', text).strip ()
    # Remove all digits
    processed_line = re.sub (r'\w*\d\w*', '', processed_line).strip ()
    return processed_line


def read_train_titles():
    """
        Function read the titles file and stores it in a list in the format (id, title)
        where id is the docid
    """
    titles = {}
    with open ("newdata/train_v_title", "r") as fp:
        for line in fp:
            id, sent = line.lower ().split ("\t")
            titles.update ({
                id: sent.strip ()
            })
    print ("Length of titles: ", len (titles))
    return titles


def read_train_keywords():
    """
        Function reads the news articles' keywords files : Thread_[0..5]
        the files contain the articles in the following formats: id_1|article_1##id_2|article_2 ....
    """
    num_files = 6
    titles = read_train_titles ()
    lines = []
    for num_file in range (num_files):
        file_path = "newdata/Thread_keywords_" + str (num_file) + ".dat"
        with open (file_path, "r") as fp:
            lines += fp.read ().split ("###")
    lines = map (lambda line: line.split ("||"), lines)
    keywords = {}
    empty = 0
    ids_set = set ()
    for line in lines:
        if len (line) >= 2:
            id, others = line[0], ' '.join (line[1:]).strip ()
            if others == "Empty":
                others = titles[id]
                empty += 1
            else:
                others = titles[id] + " " + re.sub (",", " ", others)
            keywords.update ({
                id: preprocess (others.lower ())
            })
            ids_set.add (id)
    # print ("Missing ids: ", set (titles.keys ()).difference (ids_set))

    # filter all those lines that have Empty as there article text
    return keywords


def predict(X_train, y_train, X_test):
    y_pred = []
    progress = 0
    for test_sample in X_test:
        # print ('===================================================================')
        # print ('>> ' + test_sample)
        similar_list_above_50 = []
        similar_list_below_50 = []
        for train_label, train_title in zip (y_train, X_train):
            # print(train_title)
            # print(test_sample)
            seq = difflib.SequenceMatcher (None, train_title.split (' '), test_sample.split (' '))
            similarity = seq.ratio () * 100
            if similarity > 50.0:
                similar_list_above_50.append ([train_label, similarity])
                # print (str (train_label) + " {:0.2f} ".format (similarity) + ' ' + train_title)
            else:
                similar_list_below_50.append ([train_label, similarity])
        if len (similar_list_above_50) > 0:
            top_similarities = np.array (sorted (similar_list_above_50, key=lambda t: t[1], reverse=True)[:TOP_N])
        else:
            top_similarities = np.array (sorted (similar_list_below_50, key=lambda t: t[1], reverse=True)[:TOP_N])
        top_labels = np.reshape (top_similarities[:, 0], -1)
        counter = Counter (top_labels.tolist ())
        predicted_label, majority_count = counter.most_common ()[0]
        y_pred.append (predicted_label)
        progress += 1
        print (str (progress) + '/' + str (len (X_test)))
    return y_pred


# Preparing Data
ids, keywords = zip (*read_train_keywords ().items ())
# print(keywords[0])
labels = np.reshape (np.load ('train_corrected.npy')[:, [1]], -1).tolist ()

# # Preparing Test Data
# test_df = pd.read_csv ("test_v2.csv", skiprows=[0], header=None)
# test_titles = preprocess (test_df[test_df.columns[1]])


X_train, X_test, y_train, y_test = model_selection.train_test_split (keywords, labels, test_size=0.05)
# print(len(X_test))
# print(len(y_test))

y_pred = predict (X_train, y_train, X_test)
# print(len(y_pred))
acc = accuracy_score (y_test, y_pred)
print ('Accuracy: ' + str (acc))