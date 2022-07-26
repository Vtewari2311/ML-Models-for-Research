# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import the data
dataset = pd.read_csv('')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:, 1:]

# Split dataset into training and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting multiple lineaar regression to the training set
from sklearn.linear_model import LinearRegression
regressor_new = LinearRegression()
regressor_new.fit(X_train, Y_train)
print('Model score: '+str(regressor_new.score(X_test,Y_test)))

# Predicting the test set results
y_pred = regressor.predict(X_test)

# comparing the y_prediction values with the original values because we have to calculate the accuracy of our model, which was implemented by a concept called r2_score.
# importing r2_score module
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# predicting the accuracy score
score=r2_score(y_test,y_prediction)
print(‘r2 socre is ‘,score)
print(‘mean_sqrd_error is==’,mean_squared_error(y_test,y_prediction))
print(‘root_mean_squared error of is==’,np.sqrt(mean_squared_error(y_test,y_prediction)))

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Train Set')
plt.xlabel('')
plt.ylabel('')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Test Set')
plt.xlabel('')
plt.ylabel('')
plt.show()
