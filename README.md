<1> Introduction 

code of Camel in WWW2018 paper: Camel: Content-Aware and Meta-path Augmented Metric Learning for Author Identification

Contact: Chuxu Zhang (czhang11@nd.edu)

<2> How to use

(install tensorflow, keras)

python Camel.py [parameters]

#dataset used this demo corresponds with AMiner-T data (T = 2013) in the paper

<3> Data requirement

paper-author-list-train.txt: author list of paper in training data

paper-author-list-test.txt: author list of paper in test data

paper_author_neg_ids.txt: negative author candidate of test paper for evaluation

metapath_walk.txt: random/metapath walk for indirect relation augmentation

word_embedding.txt: pre-train word embedding of paper abstract

content.pkl: paper abstract content (paper_content, paper_content_id)
