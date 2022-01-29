import numpy as np
import tensorflow as tf
import json
import random
import gzip
import math

def transfer_word(word) :
	word_list = [word]
	if (len(word) > 1 and word[-1] == 's') :
		word_list.append(word[:-1])
	if (len(word) > 4 and word[-2:] == 'es' and (word[-3] in ('s','x','z') or word[-4:-2] in ('ch','sh'))) :
		word_list.append(word[:-2])
	if (len(word) > 3 and word[-3:] == 'ies') :
		word_list.append(word[:-3] + 'y')
	if (len(word) > 3 and word[-3:] == 'ing') :
		word_list.append(word[:-3])
		word_list.append(word[:-4])
		word_list.append(word[:-3] + 'e')
	if (len(word) > 2 and word[-2:] == 'ed') :
		word_list.append(word[:-2])
		word_list.append(word[:-1])
	return word_list

class Tensorflow_data:
	def __init__(self, data_path, input_train_dir, set_name):
		#get product/user/vocabulary information
		self.product_ids = []
		with gzip.open(data_path + 'product.txt.gz', 'rt') as fin:
			for line in fin:
				self.product_ids.append(line.strip())
		self.product_size = len(self.product_ids)
		self.user_ids = []
		with gzip.open(data_path + 'users.txt.gz', 'rt') as fin:
			for line in fin:
				self.user_ids.append(line.strip())
		self.user_size = len(self.user_ids)
		self.words = []
		with gzip.open(data_path + 'vocab.txt.gz', 'rt') as fin:
			for line in fin:
				self.words.append(line.strip())

		self.vocab_size = len(self.words)
		self.query_words = []
		self.query_word_masks = []
		self.query_max_length = 0
		query_word_set = set()
		with gzip.open(input_train_dir + 'query.txt.gz', 'rt') as fin:
			for line in fin:
				words = [int(i) for i in line.strip().split(' ')]
				for word_idx in words :
					query_word_set.add(word_idx)
				if len(words) > self.query_max_length:
					self.query_max_length = len(words)
				self.query_words.append(words)
		#pad
		for i in range(len(self.query_words)):
			self.query_word_masks.append([0.0] * (self.query_max_length-len(self.query_words[i])) + [1.0] * len(self.query_words[i]))
		for i in range(len(self.query_words)):
			self.query_words[i] = [self.vocab_size for j in range(self.query_max_length-len(self.query_words[i]))] + self.query_words[i]
		#print(self.query_words)
		#print(self.query_word_masks)

		#get review sets
		self.word_count = 0
		self.vocab_distribute = np.zeros(self.vocab_size) 
		self.review_info = []
		self.review_text = []
		with gzip.open(input_train_dir + set_name + '.txt.gz', 'rt') as fin:
			for line in fin:
				arr = line.strip().split('\t')
				self.review_info.append((int(arr[0]), int(arr[1]))) # (user_idx, product_idx)
				self.review_text.append([int(i) for i in arr[2].split(' ')])
				for idx in self.review_text[-1]:
					self.vocab_distribute[idx] += 1
				self.word_count += len(self.review_text[-1])
		self.review_size = len(self.review_info)
		self.vocab_distribute = self.vocab_distribute.tolist() 
		self.sub_sampling_rate = None
		self.review_distribute = np.ones(self.review_size).tolist()
		self.product_distribute = np.ones(self.product_size).tolist()

		#get product query sets
		self.product_query_idx = []
		with gzip.open(input_train_dir + set_name + '_query_idx.txt.gz', 'rt') as fin:
			for line in fin:
				arr = line.strip().split(' ')
				query_idx = []
				for idx in arr:
					if len(idx) < 1:
						continue
					query_idx.append(int(idx))
				self.product_query_idx.append(query_idx)

		# get knowledge
		self.related_product_ids = []
		with gzip.open(data_path + 'related_product.txt.gz', 'rt') as fin:
			for line in fin:
				self.related_product_ids.append(line.strip())
		self.related_product_size = len(self.related_product_ids)
		self.brand_ids = []
		with gzip.open(data_path + 'brand.txt.gz', 'rt') as fin:
			for line in fin:
				self.brand_ids.append(line.strip())
		self.brand_size = len(self.brand_ids)
		self.category_ids = []
		with gzip.open(data_path + 'category.txt.gz', 'rt') as fin:
			for line in fin:
				self.category_ids.append(line.strip())
		self.category_size = len(self.category_ids)
	
		valid = 0
		invalid = 0		
		invalid_set = set()
		self.category_words = []
		self.category_word_masks = []
		self.category_max_length = 0
		for i in range(self.category_size) :
			words_list = self.category_ids[i].split(' ')
			if (len(words_list) > self.category_max_length) :
				self.category_max_length = len(words_list)
			words_idx_list = []
			for word in words_list :
				word = word.lower()
				word = ''.join(filter(str.isalnum, word))
				word_list = transfer_word(word)
				flag = False
				for stem_word in word_list :
					if (stem_word in self.words and self.words.index(stem_word) in query_word_set) :
						#print(stem_word, 1)
						words_idx_list.append(self.words.index(stem_word))
						valid += 1
						flag = True
						break
				if flag == False :
					for stem_word in word_list :
						if (stem_word in self.words) :
							#print(stem_word, 2)
							words_idx_list.append(self.words.index(stem_word))
							valid += 1
							flag = True
							break
				if flag == False :
					words_idx_list.append(self.vocab_size)
					invalid += 1
					invalid_set.add(word)
			self.category_words.append(words_idx_list)
		#pad
		for i in range(len(self.category_words)):
			self.category_word_masks.append([0.0] * (self.category_max_length-len(self.category_words[i])) + [1.0] * len(self.category_words[i]))
		for i in range(len(self.category_words)):
			self.category_words[i] = [self.vocab_size for j in range(self.category_max_length-len(self.category_words[i]))] + self.category_words[i]

		self.category_words = tf.constant(np.array(self.category_words))
		self.category_word_masks = tf.constant(np.array(self.category_word_masks),dtype=tf.float32)


		
		self.entity_vocab = {
			'user' : self.user_ids,
			'word' : self.words,
			'product' : self.product_ids,
			'related_product' : self.related_product_ids,
			'brand' : self.brand_ids,
			'categories' : self.category_ids
		}
		knowledge_file_dict = {
			'also_bought' : data_path + 'also_bought_p_p.txt.gz',
			'also_viewed' : data_path + 'also_viewed_p_p.txt.gz',
			'bought_together' : data_path + 'bought_together_p_p.txt.gz',
			'brand' : data_path + 'brand_p_b.txt.gz',
			'categories' : data_path + 'category_p_c.txt.gz'
		}
		knowledge_vocab = {
			'also_bought' : self.related_product_ids,
			'also_viewed' : self.related_product_ids,
			'bought_together' : self.related_product_ids,
			'brand' : self.brand_ids,
			'categories' : self.category_ids
		}

		self.product_cate_ids = []
		with gzip.open(data_path + 'category_p_c.txt.gz', 'rt') as fin:
			for line in fin :
				arr = line.strip().split(' ')
				self.product_cate_ids.append(int(arr[-1]))#min(1, len(arr) - 1)
		self.product_cate_ids.append(0)

		self.knowledge = {}
		for name in knowledge_file_dict:
			self.knowledge[name] = {}
			self.knowledge[name]['data'] = []
			self.knowledge[name]['vocab'] = knowledge_vocab[name]
			self.knowledge[name]['distribute'] = np.zeros(len(self.knowledge[name]['vocab']))
			with gzip.open(knowledge_file_dict[name], 'rt') as fin:
				for line in fin:
					knowledge = []
					arr = line.strip().split(' ')
					for x in arr:
						if len(x) > 0:
							x = int(x)
							knowledge.append(x)
							self.knowledge[name]['distribute'][x] += 1
					self.knowledge[name]['data'].append(knowledge)
			self.knowledge[name]['distribute'] = self.knowledge[name]['distribute'].tolist()


		print("Data statistic: vocab %d, review %d, user %d, product %d\n" % (self.vocab_size, 
					self.review_size, self.user_size, self.product_size))

	def sub_sampling(self, subsample_threshold):
		if subsample_threshold == 0.0:
			return
		self.sub_sampling_rate = [1.0 for _ in range(self.vocab_size)]
		threshold = sum(self.vocab_distribute) * subsample_threshold
		count_sub_sample = 0
		for i in range(self.vocab_size):
			#vocab_distribute[i] could be zero if the word does not appear in the training set
			self.sub_sampling_rate[i] = min((np.sqrt(float(self.vocab_distribute[i]) / threshold) + 1) * threshold / float(self.vocab_distribute[i]),
											1.0)
			count_sub_sample += 1

	def read_train_product_ids(self, data_path):
		self.user_train_product_set_list = [set() for i in range(self.user_size)]
		self.train_review_size = 0
		with gzip.open(data_path + 'train.txt.gz', 'rt') as fin:
			for line in fin:
				self.train_review_size += 1
				arr = line.strip().split('\t')
				self.user_train_product_set_list[int(arr[0])].add(int(arr[1]))


	def compute_test_product_ranklist(self, u_idx, original_scores, sorted_product_idxs, rank_cutoff):
		product_rank_list = []
		product_rank_scores = []
		rank = 0
		for product_idx in sorted_product_idxs:
			if product_idx in self.user_train_product_set_list[u_idx] or math.isnan(original_scores[product_idx]):
				continue
			product_rank_list.append(product_idx)
			product_rank_scores.append(original_scores[product_idx])
			rank += 1
			if rank == rank_cutoff:
				break
		return product_rank_list, product_rank_scores

	def output_ranklist(self, user_ranklist_map, user_ranklist_score_map, output_path, similarity_func):
		with open(output_path + 'test.'+similarity_func+'.ranklist', 'w') as rank_fout:
			for uq_pair in user_ranklist_map:
				user_id = self.user_ids[uq_pair[0]]
				rank = 1
				for i in range(len(user_ranklist_map[uq_pair])):
					if user_ranklist_map[uq_pair][i] >= len(self.product_ids):
						continue
					product_id = self.product_ids[user_ranklist_map[uq_pair][i]]
					rank_fout.write(user_id+'_'+str(uq_pair[1]) + ' Q0 ' + product_id + ' ' + str(rank)
							+ ' ' + str(user_ranklist_score_map[uq_pair][i])+ ' ProductSearchEmbedding\n')
					rank += 1


	def output_embedding(self, embeddings, output_file_name):
		with open(output_file_name,'w') as emb_fout:
			try:
				length = len(embeddings)
				if length < 1:
					return
				dimensions = len(embeddings[0])
				emb_fout.write(str(length) + '\n')
				emb_fout.write(str(dimensions) + '\n')
				for i in range(length):
					for j in range(dimensions):
						emb_fout.write(str(embeddings[i][j]) + ' ')
					emb_fout.write('\n')
			except:
				if isinstance(embeddings.tolist(), float):
					emb_fout.write(str(embeddings) + '\n')
				else:
					emb_fout.write(' '.join([str(x) for x in embeddings.tolist()]) + '\n')
				#else:
				#	emb_fout.write(str(embeddings) + '\n')

	def print_entity_list(self, relation_name, entity_name, entity_scores, rank_cut, remove_map):
		if entity_name not in self.entity_vocab:
			print('Cannot find entity %s' % entity_name)
		print('%s list: rank, id, name, score' % relation_name)
		sorted_entity_idxs = sorted(range(len(entity_scores)), 
									key=lambda k: entity_scores[k], reverse=True)
		entity_rank_list = []
		entity_rank_scores = []
		rank = 0
		for entity_idx in sorted_entity_idxs:
			if entity_name in remove_map and entity_idx in remove_map[entity_name]:
				continue
			if math.isnan(entity_scores[entity_idx]):
				continue
			entity_rank_list.append(entity_idx)
			entity_rank_scores.append(entity_scores[entity_idx])
			rank += 1
			if rank >= rank_cut:
				break
		# print results
		for i in range(len(entity_rank_list)):
			print('%d\t%d\t"%s"\t%.4f' % (i, entity_rank_list[i], 
				self.entity_vocab[entity_name][entity_rank_list[i]], entity_rank_scores[i]))
	
	def get_idx(self, input_str, entity_name):
		if entity_name not in self.entity_vocab:
			print('Cannot find entity %s' % entity_name)
			return None
		try:
			return self.entity_vocab[entity_name].index(input_str)
		except:
			return int(input_str)



