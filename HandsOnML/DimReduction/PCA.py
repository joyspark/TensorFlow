"""
CHAPTER8 차원 축소 (p273~)

[참고 자료]
https://nbviewer.jupyter.org/github/rickiepark/handson-ml2/blob/master/08_dimensionality_reduction.ipynb
https://onlytojay.medium.com/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EA%B0%80%EC%9E%A5-%EC%A7%A7%EC%9D%80-pca-%EC%BD%94%EB%93%9C-667f13ff3e47

"""
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_swiss_roll

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


import matplotlib.pyplot as plt

import numpy as np

# 데이터 받아오기
cancer = load_breast_cancer()
X = cancer.data
print('===== Breast Cancer Datasets ======')
print('-- X shape => ', X.shape)
# print(X)

# 데이터 표준화
X_ = StandardScaler().fit_transform(X)
print('===== Data Standardization =====')
print('-- X_ shape => ', X_.shape)
# print(X_)

"""
[Part1] 기본적인 사용 방법
"""

# PCA(2D)
pca = PCA(n_components=2)
pc = pca.fit_transform(X_)
print('===== Data Dim Reduction by PCA (dim : 30 > 2) =====')
print('-- pc shape => ', pc.shape)
print(pc)

# 차원 축소된 데이터 시각화
# plt.scatter(pc[:, 0], pc[:, 1])
# plt.show()

# 설명된 분산의 비율
# 데이터가 다 준비 되면.. %에 따라 세번째, 네번째 PC 에 정보량이 어떤지 보고 축소할 차원 수를 선택해도 될것 같음
print('===== PC에 따른 분산 비율 =====')
print(pca.explained_variance_ratio_)  # 44.27%가 첫 번째 PC, 18.97%가 두번 째 PC를 따라 놓여 있음

# 적절한 차원 수 선택 (분산 95% 유지하는데 필요한 차원 수 계산)
pca = PCA()
pca.fit(X_)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print('- cumsum => ', cumsum)
d = np.argmax(cumsum >= 0.95) + 1
print('- 분산 95% 이상 되는 차원 수 => ', d)

# 분산 % 로 차원 축소하는 방법
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_)
print('===== Data Dim Reduction by PCA (95%) =====')
print('-- X_reduced shape => ', X_reduced.shape)
print(X_reduced)

"""
[Part2] 압축 --> 복원 
원본과 똑같은 복원은 안되지만, 거의 유사한 복원이 가능하다.
ex. MNIST 를 분상 95% 유지한다면 feature가 784개에서 약 150개 정도로 줄어들 것
-> 대부분의 분산을 유지하면서도, 원본의 크기 20% 미만이되었음
-> 상당한 압축으로 알고리즘의 속도를 크게 높일 수 있다.
"""
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X_)
X_recovered = pca.inverse_transform(X_reduced)  # 복원
print('===== Reduce & Recovered =====')
print('-- X_reduced shape => ', X_reduced.shape)
print('-- X_recovered shape => ', X_recovered.shape)

"""
[Part3] 기타 PCA 방법
"""
# 랜덤 PCA (svd_solver default value = auto)
rnd_pca = PCA(n_components=10, svd_solver="randomized")
X_reduced = rnd_pca.fit_transform(X_)
print('===== Random PCA =====')
print('-- X_reduced shape => ', X_reduced.shape)

# 점진적 PCA (전체 데이터를 메모리에 올려야하는 문제를 미니배치를 이용함)
# 훈련 데이터 세트가 클때, 새로운 데이터가 준비되는 대로 실시간 PCA를 적용하는데 유용
n_batches = 50
inc_pca = IncrementalPCA(n_components=10)
for X_batch in np.array_split(X_, n_batches):
    inc_pca.partial_fit(X_batch)
X_reduced = inc_pca.transform(X_)
print('===== Incremental PCA =====')
print('-- X_reduced shape => ', X_reduced.shape)

"""
[Part4] 커널 PCA
: 샘플 군집 유지, 꼬인 매니폴드에 가까운 데이터 셋 펼치는데 유용
: 여러가지 커널 종류가 있음
"""
# 데이터 생성
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.04)
y = t > 6.9
X_reduced = rbf_pca.fit_transform(X)
print('===== Kernel PCA =====')
print('-- X_reduced shape => ', X_reduced.shape)

# 커널 선택과 하이퍼 파라미터 튜닝 (kPCA는 비지도 학습으로 명확한 성능 측정은 없음)
# 그리드 탐색 이용
clf = Pipeline([
        ("kpca", KernelPCA(n_components=2)),
        ("log_reg", LogisticRegression(solver="lbfgs"))
    ])

param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ["rbf", "sigmoid"]
    }]

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)
print('-- best_params_ => ', grid_search.best_params_)

rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.0433,
                    fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)
print('-- mse => ', mean_squared_error(X, X_preimage))
