import re
import difflib
import pandas as pd
import numpy as np
from bow import complete_preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble


def read_titles(title):
    """
        Function read the titles file and stores it in a list in the format (id, title)
        where id is the docid
    """
    titles = {}
    with open(title, "r") as fp:
        for line in fp:
            id, sent = line.lower().split("\t")
            titles.update({
                id: sent.strip()
            })
    print("Length of titles: ", len(titles))
    return titles


def read_news_articles(fn, title, num_f):
    """
        Function reads the news articles files : Thread_[0..3]
        the files contain the articles in the following formats: id_1|article_1##id_2|article_2 ....
    """
    num_files = num_f
    titles =  read_titles(title)
    lines = []
    for num_file in range(num_files):
        file_path = fn + str(num_file) + ".dat"
        with open(file_path, "r") as fp:
            lines += fp.read().split("###")
    lines = map(lambda line: line.split("||"), lines)
    articles  = {}
    empty = 0
    ids_set = set()
    for line in lines:
        if len(line) >= 2:
            id, others = line[0], ' '.join(line[1:]).strip()
            if others == "Empty":
                others = titles[id]
                empty += 1
            else:
                others = titles[id] + " " + others
                #others = titles[id]
            articles.update({
                id: join_lines(complete_preprocessing(others.lower().split("\n")))
            })
            ids_set.add(id)

    #filter all those lines that have Empty as there article text
    return articles

def join_lines(lines):
    sentence = []
    for line in lines:
        sentence += line
    return " ".join(list(set(sentence)))

def read_labels(cat_path):
    category = {}
    with open(cat_path, "r") as fp:
        for line in fp:
            id, sent = line.lower().split("\t")
            category.update({
                id: int(sent.strip())
            })
    return category

def pred(train_title_fp,
        test_title_fp,
        train_dir,
        test_dir,
        cat_path
        ):
    titles_map, ttitles_map = read_titles(train_title_fp), read_titles(test_title_fp)
    ids, articles = zip(*read_news_articles(train_dir + "Thread_keywords_", train_title_fp, 6).items())
    tids, tarticles = zip(*read_news_articles(test_dir + "Thread_keywords_", test_title_fp, 6).items())
    cat_dict = read_labels(cat_path)
    categories = []

    """
    #modify categories
    corrected_categories = np.load("../train_corrected.npy")
    corrected_id_to_label_map = {}
    for i in corrected_categories:
        id, label = str(i[0]), str(i[1])
        corrected_id_to_label_map.update({
            id: label
        })

    print("Corrected labels map: ", corrected_id_to_label_map)
    """
    titles, ttitles = [], []

    for id in ids:
        categories.append(cat_dict[id])
        #categories.append(corrected_id_to_label_map[id])
        titles.append(titles_map[id])
    for id in tids:
        ttitles.append(ttitles_map[id])

    new_articles = []
    new_categories = []


    for article, label in zip(articles, categories):
        new_categories.append(label)
        new_articles.append(article)


    categories = new_categories

    new_articles = np.array(new_articles)
    test_articles = np.array(tarticles)
    print("Create tfidf vector...")
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words='english',  max_features=750)
    tfidf_vect.fit(new_articles)

    xtrain_tfidf =  tfidf_vect.transform(new_articles)
    xtest_tfidf = tfidf_vect.transform(test_articles)

    predictions = []

    def get_pred(similarity_scores):
        scores_greater_than_45 = filter(lambda x: x[0] > 0.45, similarity_scores)
        scores_less_than_45 = filter(lambda x: x[0] < 0.45, similarity_scores)
        if len(scores_greater_than_45): #>= 0.20 * len(similarity_scores):
            counts = {}
            for score in scores_greater_than_45:
                if score[1] not in counts:
                    counts[score[1]] = 1
                else:
                    counts[score[1]] += 1
            return sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]
        else:
            return scores_less_than_45[0][1]

    from sklearn.metrics.pairwise import cosine_similarity
    print("Start prediction")
    for tidx, test_idf in enumerate(xtest_tfidf):
        similarity_scores = []
        for idx, train_idf in enumerate(xtrain_tfidf):
            score = cosine_similarity(test_idf, train_idf)
            similarity_scores.append((score, categories[idx], titles[idx]))
        pred = get_pred(sorted(similarity_scores, reverse=True)[0:15])
        predictions.append(np.array([tids[tidx],pred]))

    np.save("exp1_new.np", np.array(predictions))

    print("New Preds saved")


if __name__ == "__main__":
    train_dir, test_dir = "dir/", "test_dir/"
    train_title_fp, test_title_fp = train_dir + "train_v2_title", test_dir+"test_title"
    cat_path = train_dir + "train_v2_category"
    pred(train_title_fp, test_title_fp, train_dir, test_dir, cat_path)
