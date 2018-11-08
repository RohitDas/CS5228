import csv
import numpy as np

pred_lables = np.load('seq_matcher_submission_n_5_old.npy')
with open('seq_matcher_submission_n_5_old.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['article_id', 'category'])
    id = 1
    for label in pred_lables:
        writer.writerow([id, int(label)])
        id += 1