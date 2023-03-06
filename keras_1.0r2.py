from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# 데이터셋 생성 (임의의 값)
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([3, 7, 11, 15])

# 모델 학습
model = LinearRegression()
model.fit(X, y)

# 예측값 계산
y_pred = model.predict(X)

# r2 score 계산
r2 = r2_score(y, y_pred)

print("r2 score:", r2)