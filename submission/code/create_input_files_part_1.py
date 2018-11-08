import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib


def create_title_file(df, output_path):
	title_file = open(output_path, "w")
	for index, row in train_df.iterrows():
        	id, title = str(row["article_id"]).strip(), row["title"].strip()
        	title_file.write("\t".join([id, title]))
        	title_file.write("\n")
        title_file.close()

def create_url_file(df, output_path):
	url_file = open(output_path, "w")
	for index, row in train_df.iterrows():
        	id, url = str(row["article_id"]).strip(), row["url"].strip()
        	url_file.write("\t".join([id, url]))
        	url_file.write("\n")
        url_file.close()


def create_category_file(df, output_path):
	category_file = open(output_path, "w")
	for index, row in train_df.iterrows():
        	id, category = str(row["article_id"]).strip(), str(row["category"])
        	category_file.write("\t".join([id, category]))
        	category_file.write("\n")
        category_file.close()

if __name__ == "__main__":

	train_file = "../data/train.csv"
	train_df = pd.read_csv(train_file)
	create_title_file(train_df, "dir/trainv1_title")
	create_url_file(train_df, "dir/trainv1_url")
	create_category_file(train_df, "dir/trainv1_category")


