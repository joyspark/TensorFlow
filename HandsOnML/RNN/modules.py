"""
범용적으로 사용되는 메소드 모음
"""
import numpy as np


# 시퀀스 가라 데이터 생성
def generate_time_serise(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)

    serise = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # sin 곡선 1
    serise += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # + sin 곡선 2
    serise += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)  # + 잡음 (noise)

    return serise[..., np.newaxis].astype(np.float32)

