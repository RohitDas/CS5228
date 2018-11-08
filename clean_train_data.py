import re
import difflib
import pandas as pd
import numpy as np
from collections import Counter


def preprocess(texts):
    processed_text = []
    for text in texts:
        # Remove non-alphanumeric characters
        processed_line = re.sub (r'\W+', ' ', text).strip ()
        # Remove all digits
        processed_line = re.sub (r'\w*\d\w*', '', processed_line).strip ()
        # Lower
        processed_line = processed_line.lower ()
        processed_text.append (processed_line)
        # print(processed_line)
    return np.array (processed_text)


train_df = pd.read_csv ("train_v2.csv", skiprows=[0], header=None)
train_np = np.array (train_df.iloc[:, [0, 6]])
train_titles = preprocess (train_df[train_df.columns[1]])
train_labels = np.array (train_df[train_df.columns[6]])
train_ids = np.array (train_df[train_df.columns[0]])

# print (train_np)

train_titles_20 = train_titles[0:100]
train_labels_20 = train_labels[0:100]
train_ids_20 = train_ids[0:100]

total_update_count = 0
progress = 0
for sample_id, sample_l, sample in zip (train_ids, train_labels, train_titles):
    # print('===================================================================')
    # print('>> ' + str(sample_l) + ' >> ' + sample)
    similar_list = []
    progress += 1
    for train_id, train_label, train_title in zip (train_ids, train_labels, train_titles):
        if sample_id != train_id:
            seq = difflib.SequenceMatcher (None, train_title, sample)
            similarity = float (seq.ratio () * 100)
            if similarity > 50.0:
                # print("%d - %.2f - %s" % (train_label, similarity, train_title))
                similar_list.append ([train_label, similarity])
    # print(similar_list)
    if len(similar_list) == 0:
        continue
    similar_list = np.array (similar_list)
    similar_list_of_labels = np.reshape (similar_list[:, 0], -1)
    # print(similar_list_of_labels.shape)

    # Majority Voting
    counter = Counter (similar_list_of_labels.tolist())
    majorirty_label, majority_count = counter.most_common ()[0]

    if sample_l != int(majorirty_label):
        print(train_np[sample_id - 1, 0])
        print (str(sample_id) + ': ' + str (train_np[sample_id - 1, 1]) + '  <--  ' + str (int(majorirty_label)))
        total_update_count += 1
        train_np[sample_id - 1, 1] = int(majorirty_label)
    print(str(progress)+'/'+str(train_np.shape[0]))

np.save('train_corrected.npy', train_np)