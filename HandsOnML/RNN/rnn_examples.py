import modules
import models

import numpy as np
from tensorflow import keras

n_steps = 50


# 사용법 예제
def ex1():
    # 데이터 생성
    serise = modules.generate_time_serise(10000, n_steps + 1)
    print('-- serise shape : ', serise.shape)
    X_train, y_train = serise[:7000, :n_steps], serise[:7000, -1]
    X_valid, y_valid = serise[7000:9000, :n_steps], serise[7000:9000, -1]
    X_test, y_test = serise[9000:, :n_steps], serise[9000:, -1]
    print('-- X_train => ', X_train.shape, ', y_train shape => ', y_train.shape)
    print('-- X_valid => ', X_valid.shape, ', y_valid shape => ', y_valid.shape)
    print('-- X_test => ', X_test.shape, ', y_test shape => ', y_test.shape)

    # 모델 생성
    m1 = models.only_simple_rnn()
    print('[model 1]')
    m1.summary()

    # simple rnn with dense layer
    m2 = models.simple_rnn()
    print('[model 2]')
    m2.summary()


# 마지막 layer가 10개 vector 값 예측
def ex2():
    m = models.simple_rnn(1)
    print('[model]')
    m.summary()
    serise = modules.generate_time_serise(1, n_steps + 10)
    X_new, Y_new = serise[:, :n_steps], serise[:, n_steps:]
    X = X_new
    for step_ahead in range(10):
        y_pred_one = m.predict(X[:, step_ahead:])[:, np.newaxis, :]
        X = np.concatenate([X, y_pred_one], axis=1)

    Y_pred = X[:, n_steps:]
    print('-- Y_pred => ', Y_pred)


def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


# seq to seq 예제 : 모든 timestep에서 예측 > 모든 timestep에서 오차 그래디언트가 흐르게 된다
# > 훈련이 안정적이고 속도를 높일 수 있다.
def ex3():
    serise = modules.generate_time_serise(1, n_steps + 10)
    Y = np.empty((10000, n_steps, 10))
    for step_ahead in range(1, 10+1):
        Y[:, :, step_ahead - 1] = serise[:, step_ahead:step_ahead+n_steps, 0]
    Y_train = Y[:7000]
    Y_valid = Y[7000:9000]
    Y_test = Y[9000:]

    m = models.simple_rnn_2(10)

    optimizer = keras.optimizers.Adam(lr=0.01)
    m.compile(loss="mse", optimizer=optimizer, metrics=[last_time_step_mse])


# 시작 메소드
if __name__ == '__main__':
    # print('---------- ex1 starts ----------')
    # ex1()
    # print('---------- ex2 starts ----------')
    # ex2()
    print('---------- ex3 starts ----------')
    ex3()
