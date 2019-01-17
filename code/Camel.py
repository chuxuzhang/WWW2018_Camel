import tensorflow as tf
import keras
from keras.preprocessing import sequence
import cPickle as pkl
import utility as U
from itertools import *
import argparse
import os

# input arguments
parser = argparse.ArgumentParser(description='demo code of Camel')

parser.add_argument('--author_num', type = int, default = 28649,
				   help = 'max id of author')

parser.add_argument('--paper_num', type = int, default = 21046,
				   help = 'max id of paper')

parser.add_argument('--embed_dim', type = int, default = 128,
				   help = 'embed dimension of author and paper')

# parser.add_argument('--hidden_n', type = int, default = 128,
#                    help = 'hidden dimension of GRU encoder')

parser.add_argument('--model_path', type=str, default='../Camel',
				   help='path to save model')

parser.add_argument('--window', type = int, default = 6,
				   help = 'window size for indirect relation')

parser.add_argument('--c_len', type = int, default = 100,
				   help = 'max len of paper content')

parser.add_argument('--batch_size', type = int, default = 500,
				   help = 'batch size of training')

parser.add_argument('--learn_rate', type = float, default = 0.001,
				   help = 'learning rate')

parser.add_argument('--train_iter_max', type = float, default = 1000,
				   help = 'max number of training iterations')

parser.add_argument('--save_model_freq', type = float, default = 5,
				   help = 'number of iterations to save model')

parser.add_argument('--c_reg', type = float, default = 0.001,
				   help = 'coefficient of regularization')

parser.add_argument('--margin_d', type = float, default = 0.1,
				   help = 'margin distance of augmented component')

parser.add_argument('--c_tradeoff', type = float, default = 0.1,
				   help = 'tradeoff coefficient of augmented component')

parser.add_argument('--data_path', type=str, default='../data',
				   help='path to data')

parser.add_argument('--train_test_label', type= int, default = 0,
				   help='train/test label: 0 - train, 1 - test, 2 - tf graph test/generate negative ids for evaluation')

parser.add_argument('--top_K', type= int, default = 10,
				   help='length of return list per paper in evaluation')


args = parser.parse_args()
print(args)

# parameters setting
author_n = args.author_num
paper_n = args.paper_num
top_K = args.top_K

embed_d = args.embed_dim
hidden_n = args.embed_dim
c_len = args.c_len
c_reg = args.c_reg
margin_d = args.margin_d
c_tradeoff = args.c_tradeoff

batch_s = args.batch_size
lr = args.learn_rate
iter_max = args.train_iter_max
save_freq = args.save_model_freq

data_path = args.data_path
model_path = args.model_path

train_test_label = args.train_test_label

# data preparation
input_data = U.input_data(args = args)
word_embed = input_data.word_embed

# generate negative author ids in evaluation
if train_test_label == 2:
	print "test"
	#p_text_all = input_data.p_content[input_data.test_p_id_list]
	#print len(p_text_all)
	#input_data.gen_evaluate_neg_ids()

# Camel (objective function formulation) begin #
if train_test_label == 0:
	# tensor preparation
	# direct and indirect triple relations
	p_a_a_dir = tf.placeholder(tf.int32, [None, 3])
	p_a_a_indir = tf.placeholder(tf.int32, [None, 3])
	# paper content in direct and indirect relations
	p_c_dir_input = tf.placeholder(tf.int32, [None, c_len])
	p_c_indir_input = tf.placeholder(tf.int32, [None, c_len])

	# define latent features/parameters of author
	author_embed = tf.Variable(tf.random_normal([author_n, embed_d], mean = 0, stddev = 0.01), name = "a_latent_pars")
	# pretrain word embedding of paper content
	p_c_dir_word_e = tf.cast(tf.nn.embedding_lookup(word_embed, p_c_dir_input), tf.float32)
	p_c_indir_word_e = tf.cast(tf.nn.embedding_lookup(word_embed, p_c_indir_input), tf.float32)
	# GRU encoder 
	cell = tf.contrib.rnn.GRUCell(hidden_n)
	p_c_dir_deep_e, dir_state = tf.nn.dynamic_rnn(cell, p_c_dir_word_e, dtype = tf.float32)
	with tf.variable_scope('', reuse=True):
		p_c_indir_deep_e, indir_state = tf.nn.dynamic_rnn(cell, p_c_indir_word_e, dtype = tf.float32)
	p_c_dir_e = tf.reduce_mean(p_c_dir_deep_e, 1)
	p_c_indir_e = tf.reduce_mean(p_c_indir_deep_e, 1)

	# accumuate loss
	# loss of direct relation: distance metric learning
	Loss_1 = []
	for i in range(batch_s):
		p_e = tf.gather(p_c_dir_e, i)
		a_e_pos = tf.gather(author_embed, p_a_a_dir[i][1])
		a_e_pos = tf.reshape(a_e_pos, [1, embed_d])
		a_e_neg = tf.gather(author_embed, p_a_a_dir[i][2])
		a_e_neg = tf.reshape(a_e_neg, [1, embed_d])

		#margin loss
		Loss_1.append(tf.maximum(margin_d + tf.reduce_sum(tf.square(tf.subtract(p_e, a_e_pos))) - tf.reduce_sum(tf.square(tf.subtract(p_e, a_e_neg))), tf.zeros([1, 1])))

	# loss of indirect relation: heterogeneous Skip-gram
	bias = tf.Variable(0.1, trainable = True)
	Loss_2 = []
	for i in range(batch_s):
		p_e = tf.gather(p_c_indir_e, i)
		a_e_pos = tf.gather(author_embed, p_a_a_indir[i][1])
		a_e_pos = tf.reshape(a_e_pos, [1, embed_d])
		a_e_neg = tf.gather(author_embed, p_a_a_indir[i][2])
		a_e_neg = tf.reshape(a_e_neg, [1, embed_d])

		#cross entropy loss for graph smoothness constraint
		# negative sampling degrades to cross entropy when negative size = 1
		sum1 = tf.log(tf.sigmoid(tf.reduce_sum(tf.multiply(p_e, a_e_pos)) + bias))
		sum2 = tf.log(tf.sigmoid(- tf.reduce_sum(tf.multiply(p_e, a_e_neg)) - bias))
		Loss_2.append(- (sum1 + sum2))

	# joint loss
	t_v = tf.trainable_variables()
	reg_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in t_v])
	# objective without graph smoothness constraint 
	#joint_loss = tf.reduce_sum(Loss_1) + c_reg * reg_loss
	# objective of Camel 
	joint_loss = tf.reduce_sum(Loss_1) + c_tradeoff * tf.reduce_sum(Loss_2) + c_reg * reg_loss

	# optimizer graph smoothness constraint 
	optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(joint_loss)
# Camel (objective function formulation) end #

# train/test 
if train_test_label == 0:# train model
	init = tf.global_variables_initializer()
	saver = tf.train.Saver(max_to_keep = 2)
	with tf.Session(config = tf.ConfigProto(inter_op_parallelism_threads = 2,
			intra_op_parallelism_threads = 2)) as sess:
		sess.run(init)
		for epoch in range(1, iter_max):
			print("epoch: "+str(epoch))
			p_a_a_dir_batch = input_data.p_a_a_dir_next_batch()
			p_a_a_indir_batch = input_data.p_a_a_indir_next_batch()

			mini_batch_n = int(len(p_a_a_dir_batch)/batch_s)

			# divide each iteration into some mini batches 
			for i in range(mini_batch_n):
				p_a_a_dir_mini_batch = p_a_a_dir_batch[i*batch_s:(i+1)*batch_s]
				p_c_dir_mini_batch = input_data.gen_content_mini_batch(p_a_a_dir_mini_batch)
				p_a_a_indir_mini_batch = p_a_a_indir_batch[i*batch_s:(i+1)*batch_s]
				p_c_indir_mini_batch = input_data.gen_content_mini_batch(p_a_a_indir_mini_batch)

				feed_dict = {p_a_a_dir: p_a_a_dir_mini_batch, p_c_dir_input: p_c_dir_mini_batch, \
				p_a_a_indir: p_a_a_indir_mini_batch, p_c_indir_input: p_c_indir_mini_batch}
				_, loss_v = sess.run([optimizer, joint_loss], feed_dict)

				if i == 0:
					print("loss_value: "+str(loss_v))

			# last mini batch
			p_a_a_dir_mini_batch = p_a_a_dir_batch[len(p_a_a_dir_batch) - batch_s:len(p_a_a_dir_batch)]
			p_c_dir_mini_batch = input_data.gen_content_mini_batch(p_a_a_dir_mini_batch)
			p_a_a_indir_mini_batch = p_a_a_indir_batch[len(p_a_a_indir_batch) - batch_s:len(p_a_a_indir_batch)]
			p_c_indir_mini_batch = input_data.gen_content_mini_batch(p_a_a_indir_mini_batch)

			feed_dict = {p_a_a_dir: p_a_a_dir_mini_batch, p_c_dir_input: p_c_dir_mini_batch, \
			p_a_a_indir: p_a_a_indir_mini_batch, p_c_indir_input: p_c_indir_mini_batch}
			_, loss_v = sess.run([optimizer, joint_loss], feed_dict)

			# save model for evaluation
			if epoch % save_freq == 0:
				if not os.path.exists(model_path):
					os.makedirs(model_path)
				saver.save(sess, model_path + "/Camel" + str(epoch) + ".ckpt")

				# evaluation tracking during training
				# better to batch generate paper embedding for large data
				p_text_all = input_data.p_content[input_data.test_p_id_list]
				p_text_deep_f = sess.run([p_c_dir_e], {p_c_dir_input: p_text_all})
				p_text_deep_f = p_text_deep_f[0]

				a_latent_f = tf.get_default_graph().get_tensor_by_name("a_latent_pars:0")
				a_latent_f = a_latent_f.eval()

				input_data.Camel_evaluate(p_text_deep_f, a_latent_f, top_K)

elif train_test_label == 1:# test model
	with tf.Session(config = tf.ConfigProto(inter_op_parallelism_threads = 2,
			intra_op_parallelism_threads = 2)) as sess:
		restore_idx = 20
		saver.restore(sess, model_path + "Camel" + str(restore_idx) + ".ckpt")

		# load paper semantic deep embedding by learned rnn encoder
		p_text_all = input_data.p_content
		p_text_deep_f = sess.run([p_c_dir_e], {p_c_dir_input: p_text_all})
		p_text_deep_f = p_text_deep_f[0]

		# load learned author latent features/parameters
		a_latent_f = tf.get_default_graph().get_tensor_by_name("a_latent_pars:0")
		a_latent_f = a_latent_f.eval()

		# model evaluation 
		# better to batch generate paper embedding for large data
		input_data.Camel_evaluate(p_text_deep_f, a_latent_f, top_K)
else:
	print "tf graph test finish."

