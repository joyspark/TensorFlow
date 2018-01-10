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

#********************************* 사전 작업 시작 *********************************#

twitter = Twitter()
os.environ['TF_CPP_MIN_LOG_LEVEL']='4'
today_date = datetime.date.today().strftime("%Y%m%d")

# 단어 리스트(원본 데이터에서 명사 추출 : 중복 있음)
words = []
# 명사추출 > 단어 리스트 생성 함수
def read_file(file_path):
	tmp_words = []
	contents = ""
	if not os.path.exists(file_path):
		raise Exception('file ', file_path, ' is not exists.')
	else:
		with open(file_path, 'r') as f:
			for line in f.readlines():
				tmp_words.append(twitter.nouns(line)) # 명사 추출
		tmp_words = list(itertools.chain.from_iterable(tmp_words)) # flatten
		contents = " ".join(tmp_words) # str instance로 변환
		text = tf.compat.as_str(contents)
	return text.split()

words = read_file("./data/ko_wiki.txt")
print('words : ',words[:100])
print('words length : ',len(words))

# 형태소 분석된 단어를 파일로 
try:
	with open('./dic/words.csv','w') as f :
		for i in range(len(words)) :
			if i == len(words)-1 :
				f.write(words[i])
			else :
				f.write(words[i]+',')
except OSError as e:
	if e.errno == 2:
		# 파일이나 디렉토리가 없음!
		print ('No such file or directory to remove')
		pass
	else:
		raise
# 빈도가 높은 26000개 대상으로 데이터 셋 구축 (사전)
vocabulary_size = 26000

# 원본 to 사전 인덱스 매핑, 단어 출현 빈도수로 정렬된 단어 리스트
def build_dataset(vocab, n_words):
	# 빈도수 높은 n_words 개 단어 추출
	unique = collections.Counter(vocab) # word count로 중복제거
	orders = unique.most_common(n_words-1) # count가 n_words 이상인 것들 가져옴
	count = [['UNK', -1]] # unknown keyword
	count.extend(orders)
	
	# 추출한 단어의 사전 생성 (단어:index)
	dictionary = {}
	for word,_ in count:
		dictionary[word] = len(dictionary)
	
	# 원본 데이터에 사전에 있는 index 매핑
	data = []
	for word in vocab:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0
			count[0][1] += 1
		data.append(index)
	return data,count,list(dictionary.keys())

# data : 원본 단어의 인덱스로 구성된 리스트
# count : 단어/빈도 쌍으로 구성된 리스트 (뒤에서는 사용 안함)
# ordered_words : 빈도가 높은 순으로 정렬된 단어 사전 리스트
data, count, ordered_words = build_dataset(words, vocabulary_size)
print('data : ', data[:100])
print('data length: ', len(data))

# 단어들의 index 데이터를 파일로 
try:
	with open('./dic/data.csv','w') as f :
		for i in range(len(data)) :
			if i == len(data)-1 :
				f.write(str(data[i]))
			else :
				f.write(str(data[i])+',')
except OSError as e:
	if e.errno == 2:
		# 파일이나 디렉토리가 없음!
		print ('No such file or directory to remove')
		pass
	else:
		raise

# 빈도수가 높은 단어들 사전을 파일로  
try:
	with open('./dic/ordered_words.csv','w') as f :
		for i in range(len(ordered_words)) :
			if i == len(ordered_words)-1 :
				f.write(ordered_words[i])
			else :
				f.write(ordered_words[i]+',')
except OSError as e:
	if e.errno == 2:
		# 파일이나 디렉토리가 없음!
		print ('No such file or directory to remove')
		pass
	else:
		raise
		
del words, count # 사용하지 않는 변수 삭제

#********************************* 사전작업 끝 *********************************#

# mini_batch 사용할 샘플 데이터 생성 메소드
def generate_batch(data, batch_size, num_skips, skip_window, data_index):
	assert batch_size%num_skips == 0
	assert num_skips <= 2*skip_window
	
	temp = 'batch_size {}, num_skips {}, skip_window {}, data_index {}'
	
	# 배치 크기만큼 임의의 난수 생성
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size,1), dtype=np.int32)
	
	span = 2 * skip_window + 1
	assert span > num_skips

	# data_index 번째부터 span 크기만큼 단어 인덱스 저장
	buffer = collections.deque(maxlen=span)
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)#다음 단어 인덱스로 이동.

	# skip-gram은 타겟 단어로부터 주변의 컨텍스트 단어를 예측하는 모델이다.
	# 학습하기 전에 단어들을 (target, context) 형태로 변환해 주어야 한다.
	# 바깥 루프는 batch_size // num_skips
	# 안쪽 루프는 num_skips
	# batch_size는 num_skips로 나누어 떨어지기 때문에 정확하게 batch_size만큼 반복
	for i in range(batch_size // num_skips):
		targets = list(range(span))# 1. 0부터 span-1까지의 정수로 채운 다음
		targets.pop(skip_window)# 2. skip_window번째 삭제
		np.random.shuffle(targets)#  3. 난수를 사용해서 섞는다.
		
		# batch : target 단어만 들어가고, num_skips만큼 같은 단어가 중복된다.
		# labels : target을 제외한 단어만 들어가고, num_skips만큼 중복될 수 있다.
		start = i * num_skips
		batch[start:start+num_skips] = buffer[skip_window]
		
		for j in range(num_skips):
			labels[start+j, 0] = buffer[targets[j]]
		
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
		
	data_index = (data_index + len(data) - span) % len(data)
	return batch, labels, data_index


#********************************* 학습 그래프 셋팅 *********************************#
np.random.seed(1)
tf.set_random_seed(1)

batch_size = 100        # 일반적으로 16 <= batch_size <= 512
embedding_size = 100    # embedding vector 크기
skip_window = 1         # target 양쪽의 단어 갯수
num_skips = 2           # 컨텍스트로부터 생성할 레이블 갯수

# valid_examples : [80 84 33 81 93 17 36 82 69 65 92 39 56 52 51 32]
# replace는 중복 허용 안함. 30보다 작은 정수에서 5개 고르기.
#valid_size = 12     # 유사성을 평가할 단어 집합 크기
#valid_window = 50  # 앞쪽에 있는 분포들만 뽑기 위한 샘플
#valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 30    # negative 샘플링 갯수

# valid_dataset은 valid_examples 배열의 tf 상수 배열.
train_inputs = tf.placeholder(tf.int32, shape=[batch_size], name='train_inputs')
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1], name='train_labels')
#valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
# 모델 사용시 다른 수동으로 데이터를 넣어주기 위해
# 미리 정의하는 constant 말고 실행시 정해주는 placeholder로 바꿈
valid_dataset = tf.placeholder(tf.int32, shape=[None], name='valid_dataset') # 확인해볼 단어 index

# NCE loss 변수. weights는 (50000, 128), biases는 (50000,).
truncated = tf.truncated_normal([vocabulary_size, embedding_size],
								stddev=1.0 / math.sqrt(embedding_size))
nce_weights = tf.Variable(truncated)
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# embeddings 벡터. embed는 바로 아래 있는 tf.nn.nce_loss 함수에서 단 1회 사용
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# 배치 데이터에 대해 NCE loss 평균 계산
nce_loss = tf.nn.nce_loss(weights=nce_weights,
							biases=nce_biases,
							labels=train_labels,
							inputs=embed,
							num_sampled=num_sampled, # embed 에서 랜덤하게 뽑아냄
							num_classes=vocabulary_size)
loss = tf.reduce_mean(nce_loss)

# SGD optimizer
optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

# 유사도를 계산하기 위한 모델. 학습 모델은 optimizer까지 구축한 걸로 종료.
# minibatch 데이터(valid embeddings)와 모든 embeddins 사이의 cosine 유사도 계산
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True, name='similarity')


#********************************* skip-gram 모델 학습 *********************************#

num_steps = 1000001             # 마지막 반복을 출력하기 위해 +1.

start_time = datetime.datetime.now()
train_start = start_time.strftime('%Y%m%d:%H%M%S')
print("["+train_start+"][INFO] TRAIN...[BEGINE]")

saver = tf.train.Saver()
with tf.Session() as session:
	session.run(tf.global_variables_initializer())

	average_loss, data_index = 0,0
	for step in range(num_steps):
		batch_inputs, batch_labels, data_index = generate_batch(data, batch_size, num_skips, skip_window, data_index)

		feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
		_, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
		average_loss += loss_val

		if step % 2000 == 0:
			if step > 0:
				average_loss /= 2000
			print('Average loss at step {} : {}'.format(step, average_loss))
			average_loss = 0
	#***************************** 학습 모델 저장 *****************************#
	file_path = './model'
	if(not os.path.exists(file_path)):
		os.makedirs(file_path)
	if(os.path.exists(file_path+"/embedding")):
		shutil.rmtree(file_path+"/embedding")
	builder = tf.saved_model.builder.SavedModelBuilder(file_path+"/embedding")
	builder.add_meta_graph_and_variables(session,[tf.saved_model.tag_constants.SERVING])
	builder.save(True)
	print("Java Model Saved in file :", file_path)
	if(not os.path.exists("./model")):
		os.makedirs("./model")
	save_path = saver.save(session, "./model/"+today_date+"/model.chpk")
	print("Model saved in file : ", save_path)
	
end_time = datetime.datetime.now()
train_end = end_time.strftime('%Y%m%d:%H%M%S')
print("["+train_end+"][INFO] TRAIN...[END]")
end_datetime = datetime.datetime.now()
str_end = end_datetime.strftime('%Y%m%d:%H%M%S')
print("["+str_end+"][INFO] PROGRAM...[END]")
