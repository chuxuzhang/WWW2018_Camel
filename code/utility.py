import six.moves.cPickle as pickle
#import pandas as pd
import numpy as np
import string
import re
import random
from keras.preprocessing import sequence
from itertools import *


class input_data:
	def __init__(self, args):
		self.args = args

		# direct paper-author relation
		p_a_dir_list_train = [[] for k in range(self.args.paper_num)]
		p_a_dir_list_test = [[] for k in range(self.args.paper_num)]
		author_train = [0] * self.args.author_num
		dir_relation_f = ["/paper-author-list-train.txt", "/paper-author-list-test.txt"]
		#p_a_list_train_f = open(self.args.data_path + "/paper_author_list_train.txt", "r")
		for f_index in range(len(dir_relation_f)):
			f_name = dir_relation_f[f_index]
			neigh_f = open(self.args.data_path + f_name, "r")
			for line in neigh_f:
				line = line.strip()
				p_index = string.atoi(re.split(':',line)[0])
				a_list = re.split(',',re.split(':',line)[1])
				if f_name == "/paper-author-list-train.txt":
					for i in range(len(a_list)):
						p_a_dir_list_train[p_index].append('a'+str(a_list[i]))
						author_train[int(a_list[i])] = 1
				elif f_name == "/paper-author-list-test.txt":
					for j in range(len(a_list)):
						p_a_dir_list_test[p_index].append('a'+str(a_list[j]))
			neigh_f.close()

		self.p_a_dir_list_train = p_a_dir_list_train
		self.p_a_dir_list_test = p_a_dir_list_test
		#print p_a_dir_list_test[13113]
		self.dir_len = sum(len(x) for x in self.p_a_dir_list_train)
		self.author_train = author_train

		test_p_id_list = []
		for i in range(self.args.paper_num):
			if len(p_a_dir_list_test[i]):
				test_p_id_list.append(i)

		self.test_p_id_list = test_p_id_list
		
		# indirect paper-author relation from heterogeneous walk
		p_a_indir_list_train = [[] for k in range(self.args.paper_num)]
		def p_a_indir_set(path):
			indir_relation_f = ["/APA_walk.txt", "/APPA_walk.txt", "/APVPA_walk.txt"]
			#het_walk_f = open(self.args.data_path + "/het_random_walk.txt", "r")
			for f_index in range(len(indir_relation_f)):
				f_name = indir_relation_f[f_index]
				neigh_f = open(self.args.data_path + f_name, "r")
				for line in neigh_f:
					line=line.strip()
					path = re.split(' ',line)
					for k in range(len(path)):
						curr_node = path[k]
						if curr_node[0] == 'p':
							for w in range(k - self.args.window, k + self.args.window +1):
								if w >= 0 and w < len(path) and w != k:
									neigh_node = path[w]
									if neigh_node[0] == 'a' and neigh_node not in self.p_a_dir_list_train[int(curr_node[1:])]:
										p_a_indir_list_train[int(curr_node[1:])].append(neigh_node)
				neigh_f.close()
			return p_a_indir_list_train

		self.p_a_indir_list_train = p_a_indir_set(self.args.data_path)
		self.indir_len = sum(len(x) for x in self.p_a_indir_list_train)


		def load_p_content(path, word_n = 100000):
			f = open(path, 'rb')
			p_content_set = pickle.load(f)
			f.close()

			def remove_unk(x):
				return [[1 if w >= word_n else w for w in sen] for sen in x]

			p_content, p_content_id = p_content_set
			p_content = remove_unk(p_content)
			p_content_set = (p_content, p_content_id)

			return p_content_set


		def load_word_embed(path, word_n = 54559, word_dim = 128):
			word_embed = np.zeros((word_n + 2, word_dim))

			f = open(path,'r')
			for line in islice(f, 1, None):
				index = int(line.split()[0])
				embed = np.array(line.split()[1:])
				word_embed[index] = embed

			return word_embed

		# text content (e.g., abstract) of paper and pretrain word embedding 
		self.p_content, self.p_content_id = load_p_content(path = self.args.data_path + '/content.pkl')
		self.p_content = sequence.pad_sequences(self.p_content, maxlen = self.args.c_len, value = 0., padding = 'post', truncating = 'post') 
		self.word_embed = load_word_embed(path = self.args.data_path + '/word_embedding.txt')


	def p_a_a_dir_next_batch(self):
		p_a_a_dir_list_batch = []
		for i in range(self.args.paper_num):
			for j in range(len(self.p_a_dir_list_train[i])):
				a_neg = random.randint(0, self.args.author_num - 1)
				while (('a'+str(a_neg)) in self.p_a_dir_list_train[i]):
					a_neg = random.randint(0, self.args.author_num - 1)
				a_pos = int(self.p_a_dir_list_train[i][j][1:])
				triple=[i, a_pos, a_neg]
				p_a_a_dir_list_batch.append(triple)
		return p_a_a_dir_list_batch


	def p_a_a_indir_next_batch(self):
		p_a_a_indir_list_batch = []
		p_threshold = float(self.dir_len)/self.indir_len + 3e-3
		#print p_threshold
		for i in range(self.args.paper_num):
			for j in range(len(self.p_a_indir_list_train[i])):
				if random.random() < p_threshold:
					a_neg = random.randint(0, self.args.author_num - 1)
					while (('a'+str(a_neg)) in self.p_a_dir_list_train[i]):
						a_neg = random.randint(0, self.args.author_num - 1)
					a_pos = int(self.p_a_indir_list_train[i][j][1:])
					triple=[i, a_pos, a_neg]
					p_a_a_indir_list_batch.append(triple)
		return p_a_a_indir_list_batch


	def gen_content_mini_batch(self, triple_batch):
		p_c_data = []
		for i in range(len(triple_batch)):
			c_temp = (self.p_content[triple_batch[i][0]]).reshape(self.args.c_len)
			p_c_data.append(c_temp)
		return p_c_data


	def gen_evaluate_neg_ids(self):
		#neg_num = 100
		author_n_ave = 0
		paper_n = 0
		p_a_neg_ids_f = open(self.args.data_path + "/paper_author_neg_ids.txt", "w")
		for i in range(self.args.paper_num):
			if len(self.p_a_dir_list_test[i]):
				p_a_neg_ids_f.write(str(i) + ":")
				neg_num = 100 - len(self.p_a_dir_list_test[i])
				for j in range(neg_num):
					neg_id = random.randint(0, self.args.author_num - 1)
					neg_id_str = 'a' + str(neg_id)
					while (neg_id_str in self.p_a_dir_list_test[i]):
						neg_id = random.randint(0, self.args.author_num - 1)
						neg_id_str = 'a' + str(neg_id)
					p_a_neg_ids_f.write(str(neg_id) + ",")
				p_a_neg_ids_f.write("\n")
				author_n_ave += len(self.p_a_dir_list_test[i])
				paper_n += 1
		p_a_neg_ids_f.close()
		print ("author_n_ave_test: " + str(float(author_n_ave)/paper_n))


	def Camel_evaluate(self, p_text_deep_f, a_latent_f, top_K):
		p_id_map = [0] * self.args.paper_num

		new_id_temp = 0
		for k in range(len(self.test_p_id_list)):
			p_id_temp = self.test_p_id_list[k]
			p_id_map[p_id_temp] = new_id_temp
			new_id_temp += 1

		p_a_neg_list_test = [[] for k in range(self.args.paper_num)]
		p_a_neg_ids_f = open(self.args.data_path + "/paper_author_neg_ids.txt", "r")
		for line in p_a_neg_ids_f:
			line = line.strip()
			p_id = int(re.split(':', line)[0])
			a_list = re.split(':', line)[1]
			a_list_ids = re.split(',', a_list)
			for i in range(len(a_list_ids) - 1):
				p_a_neg_list_test[p_id].append(int(a_list_ids[i]))
		p_a_neg_ids_f.close()

		# only evaluate test paper which has author in training data
		test_p_has_train_a = [0] * self.args.paper_num
		for i in range(self.args.paper_num):
			for j in range(len(self.p_a_dir_list_test[i])):
				a_id_temp = int(self.p_a_dir_list_test[i][j][1:])
				if self.author_train[a_id_temp]:
					test_p_has_train_a[i] += 1

		# Recall/Precision Scores
		recall_ave = 0
		pre_ave = 0
		evaluate_p_num = 0 
		ave_a_num = 0.0 

		for i in range(self.args.paper_num):
			if len(self.p_a_dir_list_test[i]) and len(p_a_neg_list_test[i]) and test_p_has_train_a[i]:
				evaluate_p_num += 1
				correct_num = 0
				score_list = []

				for j in range(len(self.p_a_dir_list_test[i])):
					a_id_temp = int(self.p_a_dir_list_test[i][j][1:])
					score_temp = np.dot(p_text_deep_f[p_id_map[i]], a_latent_f[a_id_temp])
					score_list.append(score_temp)

				for k in range(len(p_a_neg_list_test[i])):
					a_id_temp = p_a_neg_list_test[i][k]
					score_temp = np.dot(p_text_deep_f[p_id_map[i]], a_latent_f[a_id_temp])
					score_list.append(score_temp)

				score_list.sort()

				score_threshold = score_list[ - top_K - 1]

				for jj in range(len(self.p_a_dir_list_test[i])):
					a_id_temp = int(self.p_a_dir_list_test[i][jj][1:])
					if self.author_train[a_id_temp]:
						score_temp = np.dot(p_text_deep_f[p_id_map[i]], a_latent_f[a_id_temp])
						if score_temp > score_threshold:
							correct_num += 1

				recall_ave += float(correct_num) / test_p_has_train_a[i]
				pre_ave += float(correct_num) / top_K

				ave_a_num += test_p_has_train_a[i]

		print ("total evaluate paper number: " + str(evaluate_p_num))
		print ("average evaluate author number: " + str(ave_a_num / evaluate_p_num))
		recall_ave = recall_ave / evaluate_p_num
		pre_ave = pre_ave / evaluate_p_num
		F_1= (2 * recall_ave * pre_ave) /(recall_ave + pre_ave)
		print ("recall_ave@top_K: " + str(recall_ave))
		print ("pre_ave@top_K: " + str(pre_ave))

		# AUC Score
		AUC_ave = 0
		for i in range(self.args.paper_num):
			if len(self.p_a_dir_list_test[i]) and len(p_a_neg_list_test[i]) and test_p_has_train_a[i]:
				neg_score_list = []
				correct_num = 0
				pair_num = 0
				for k in range(len(p_a_neg_list_test[i])):
					a_id_temp = p_a_neg_list_test[i][k]
					score_temp = np.dot(p_text_deep_f[p_id_map[i]], a_latent_f[a_id_temp])
					neg_score_list.append(score_temp)

				for j in range(len(self.p_a_dir_list_test[i])):
					a_id_temp = int(self.p_a_dir_list_test[i][j][1:])
					if self.author_train[a_id_temp]:
						pos_score = np.dot(p_text_deep_f[p_id_map[i]], a_latent_f[a_id_temp])
						for jj in range(len(neg_score_list)):
							pair_num += 1
							if pos_score > neg_score_list[jj]:
								correct_num += 1

				AUC_ave += float(correct_num) / pair_num

		AUC_ave = AUC_ave / evaluate_p_num
		print ("AUC_ave: " + str(AUC_ave))


