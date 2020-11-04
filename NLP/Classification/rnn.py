# https://bioinformaticsandme.tistory.com/251

import tensorflow_datasets as tfds
import tensorflow as tf

import matplotlib.pyplot as plt

#tfds.disable_progress_bar()

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.show()


# ============> setup input pipeline 입력 데이터 연결
# 영화 리뷰 데이터 (라벨 : 긍/부정)
dataset, info = tfds.load('imdb_reviews/subwords8k',
                          with_info=True,  # return tuple
                          as_supervised=True)  # feature and label
train_dataset, test_dataset = dataset['train'], dataset['test']


# ============> 데이터 셋 info에 인코딩 정보를 가져온다.
encoder = info.features['text'].encoder
print('Vocab size : {}'.format(encoder.vocab_size))  # 8185 (컴퓨터가 인식하는 단어 수)


# --------- 인코딩 해보기 ---------
sample_str = 'Hello TensorFlow'

encoded_str = encoder.encode(sample_str)
print('Encoded Str is {}'.format(encoded_str))  # [4025, 222, 6307, 2327, 4043, 2120]

original_str = encoder.decode(encoded_str)
print('The original str : {}'.format(original_str))  # Hello TensorFlow

# 예외처리, 인코딩 > 디코딩 한 결과가 처음과 같지 않을 때 = error 발생
assert original_str == sample_str
# 인코드된 정보 확인  [Hell, o, Ten, sor, Fl, ow]
for index in encoded_str:
    print('{} ----> {}'.format(index, encoder.decode([index])))
# --------- 인코딩 해보기 ---------


# ============> 학습을 위한 데이터 준비
# 25,000 / 64 => 391 개의 batch 데이터를 준비,
BUFFER_SIZE = 10000  # 1 epoch 의 데이터 수
BATCH_SIZE = 64  # 1 step 의 데이터 수 & 동시에 max length가 64 (one-hot, embedding 하기 전 데이터)
# train 25,000 개 test 25,000개
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
# [0] input (64, 제각각), [1] output(64,)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, train_dataset.output_shapes)  # tuple((None, None), (None,))
test_dataset = test_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)  # tuple((None, None), (None,))


# ============> 모델 생성
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),  # 8185 -> 64 차원으로 표현 embedding
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),  # units = 64 (dimensionality of the output space)
    tf.keras.layers.Dense(64, activation='relu'),  # units = 64
    tf.keras.layers.Dense(1, activation='sigmoid')  # unit = 1
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


# ============> 모델 학습 (25,000/64 batch => 391 epochs)
# 1 epoch에 거의 26분소요
history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset,
                    validation_steps=30)


# ============> test set 정확도 평가
test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss : {}'.format(test_loss))
print('Test Accuracy : {}'.format(test_acc))


# padding method 모델 입력 샘플 길이를 동일하게 맞추는 작업
def pad_to_size(vec, size):
    zeros = [0]*(size-len(vec))
    vec.extend(zeros)
    return (vec)

# 예측 함수 정의 (예측 값 0.5 이상 = 긍정, 미만 = 부정)
def sample_predict(sentence, pad):
    encoded_sample_pred_text = encoder.encode(sentence)

    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))  # 0번째 축 차원 확장

    return (predictions)

# 패딩 작업 없이 sample text 분류
sample_pred_text = ('The movie was cool. The animation and the graphics were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=False)
print(predictions)

# 패딩 작업으로 Sample text 분류
sample_pred_text = ('The movie was cool. The animation and the graphics were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print(predictions)

# 모델 정확도 시각화
plot_graphs(history, 'accuracy')

# 모델 손실 시각화
plot_graphs(history, 'loss')

