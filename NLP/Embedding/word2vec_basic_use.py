# -*- coding: utf-8 -*-

# 참조 : http://pythonkim.tistory.com/93?category=613486
import tensorflow as tf
import numpy as np
import math
import shutil
import collections
import itertools
import os
import datetime
from konlpy.tag import Twitter # 트위터 형태소 분석기

start_datetime = datetime.datetime.now()
str_start = start_datetime.strftime('%Y%m%d:%H%M%S')
print("["+str_start+"][INFO] PROGRAM...[BEGINE]")

twitter = Twitter()
os.environ['TF_CPP_MIN_LOG_LEVEL']='4'
today_date = datetime.date.today().strftime("%Y%m%d")

#********************************* 사전 로드 *********************************#

words = []
# 단어 리스트 생성
try:
	with open('./dic/words.csv','r') as f :
		for word in f.readline().split(',') :
			words.append(word)
except OSError as e:
	if e.errno == 2:
		# 파일이나 디렉토리가 없음!
		print ('No such file or directory to remove')
		pass
	else:
		raise
print('words : ',words[:100])
print('words length : ',len(words))

data = []
# 단어 인텍스 리스트 생성
try:
	with open('./dic/data.csv','r') as f :
		for idx in f.readline().split(',') :
			data.append(int(idx))
except OSError as e:
	if e.errno == 2:
		# 파일이나 디렉토리가 없음!
		print ('No such file or directory to remove')
		pass
	else:
		raise
print('data : ',data[:100])
print('data length : ',len(data))


ordered_words_dic = {}
# 빈도수가 높은 단어 사전 생성{단어 : 인덱스}
try:
	with open('./dic/ordered_words.csv','r') as f :
		words = f.readline().split(',')
		for i in range(len(words)) :
			ordered_words_dic[words[i]] = i
except OSError as e:
	if e.errno == 2:
		# 파일이나 디렉토리가 없음!
		print ('No such file or directory to remove')
		pass
	else:
		raise
print('dataordered_words_dic length : ',len(ordered_words_dic))

ordered_words = []
# 빈도수가 높은 단어 사전 리스트 생성 [단어]
try:
	with open('./dic/ordered_words.csv','r') as f :
		for word in f.readline().split(',') :
			ordered_words.append(word)
except OSError as e:
	if e.errno == 2:
		# 파일이나 디렉토리가 없음!
		print ('No such file or directory to remove')
		pass
	else:
		raise
print('ordered_words : ',ordered_words[:100])
print('dataordered_words length : ',len(ordered_words))

vocabulary_size = len(ordered_words) #26000

#********************************* 학습 그래프 셋팅 *********************************#
np.random.seed(1)
tf.set_random_seed(1)

batch_size = 100        # 일반적으로 16 <= batch_size <= 512
embedding_size = 100    # embedding vector 크기
skip_window = 1         # target 양쪽의 단어 갯수
num_skips = 2           # 컨텍스트로부터 생성할 레이블 갯수

# valid_examples : [80 84 33 81 93 17 36 82 69 65 92 39 56 52 51 32]
# replace는 중복 허용 안함. 30보다 작은 정수에서 5개 고르기.
valid_size = 15     # 유사성을 평가할 단어 집합 크기
valid_window = 100  # 앞쪽에 있는 분포들만 뽑기 위한 샘플
valid_dataset = tf.placeholder(tf.int32, shape=[None], name='valid_dataset') # 확인해볼 단어 index
num_sampled = 30    # negative 샘플링 갯수

# valid_dataset은 valid_examples 배열의 tf 상수 배열.
train_inputs = tf.placeholder(tf.int32, shape=[batch_size], name='train_inputs')
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1], name='train_labels')

# embeddings 벡터. embed는 바로 아래 있는 tf.nn.nce_loss 함수에서 단 1회 사용
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

# 유사도를 계산하기 위한 모델. 학습 모델은 optimizer까지 구축한 걸로 종료.
# minibatch 데이터(valid embeddings)와 모든 embeddins 사이의 cosine 유사도 계산
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True, name='similarity')


#********************************* 임베딩 사용 *********************************#
num_steps = 1000001             # 마지막 반복을 출력하기 위해 +1.

start_time = datetime.datetime.now()
use_start = start_time.strftime('%Y%m%d:%H%M%S')
print(use_start)

saver = tf.train.Saver()

with tf.Session() as session:
	session.run(tf.global_variables_initializer())

#	if not os.path.exists("./model/"+today_date+"/model.chpk"):
	model_path="./model/"+today_date+"/model.chpk"

	try :
		saver.restore(session, model_path)
	except OSError as e:
		if e.errno == 2:
		# 파일이나 디렉토리가 없음!
			print ('No such file or directory to remove')
			pass
		else:
			raise

	test_words = np.array(['경제학','예술']) # word : 테스트 데이터
	test_data = [] # word > 사전 index list로 변환
	for i in range(len(test_words)):
		test_data.append(ordered_words_dic[test_words[i]])
	valid_examples_ = np.array(test_data).astype(np.int32) # numpy array로 만들어줌

	for i in range(len(test_words)):
		valid_word = test_words[i]
		sim = session.run(similarity, feed_dict = {valid_dataset:valid_examples_})
		print('valid_examples_ : ',valid_examples_)
		print('valid_examples_ shape : ',valid_examples_.shape)
		
		top_k = 5
		nearest = sim[i].argsort()[-top_k - 1:-1][::-1] # index 0은 자기 자신이기 때문에 1부터 8까지
		print('sim shape : ', sim.shape)
		print('sim : ', sim)
		print('nearest shape : ', nearest.shape)
		print('nearest : ', nearest)
		log_str = ', '.join([ordered_words[k] for k in nearest])
		print('Nearest to {}: {}'.format(valid_word, log_str))

end_time = datetime.datetime.now()
use_end = end_time.strftime('%Y%m%d:%H%M%S')
print("["+use_end+"][INFO] USE...[END]")

end_datetime = datetime.datetime.now()
str_end = end_datetime.strftime('%Y%m%d:%H%M%S')
print("["+str_end+"][INFO] PROGRAM...[END]")
