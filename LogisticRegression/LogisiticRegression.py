# importing libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

data = np.genfromtxt(filename, delimiter=',')

label2_filter = data[:, -1] == 2
data[:, -1] = 0
data[label2_filter] = 1

X = data[:, :-1]
y = data[:, -1]

print(f'{X.shape = }')
print(f'{y.shape = }')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f'{X_train.shape = }')
print(f'{y_train.shape = }')
print(f'{X_test.shape = }')
print(f'{y_test.shape = }')

model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test) * 100
print(f'accuracy = {accuracy:.3f}%')

