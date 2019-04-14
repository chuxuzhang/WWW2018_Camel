<1> Introduction 

code of Camel in WWW2018 paper: Camel: Content-Aware and Meta-path Augmented Metric Learning for Author Identification

Contact: Chuxu Zhang (czhang11@nd.edu)


<2> How to use

(install tensorflow, keras, de-compress word_embedding.txt.zip)

python Camel.py [parameters]

#dataset used this demo corresponds with AMiner-T data (T = 2012) in the paper


<3> Data requirement

paper-author-list-train.txt: author list of paper in training data

paper-author-list-test.txt: author list of paper in test data

paper_author_neg_ids.txt: negative author candidate of test paper for evaluation

metapath_walk.txt: random/metapath walk for indirect relation augmentation

word_embedding.txt: pre-train word embedding of paper abstract

content.pkl: paper abstract content (paper_content, paper_content_id)


<4> use data_process.py to generate related data for Camel

small_data_with_map_id.txt: AMiner-T raw data with new (author, paper, venue) map id

find original data from: https://aminer.org/citation


<5> use word2vec.py to generate word embedding of paper content


