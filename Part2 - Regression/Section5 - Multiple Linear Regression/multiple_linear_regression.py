# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap/ redundant dependancies
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling | this is not needed because the libraries we use here, do it for us
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

###############################################################################
###############################################################################

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
"""
this "statsmodels" library does not take into consideration the constant b0, so we need to add an extra column of 1s.
This means that this extra column would represent the "X0" in the "b0*X0" part of the equation.
"""
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
np.set_printoptions(precision=1, suppress=True)
print(X)



"""
X_opt = X[:, [0, 1, 2, 3, 4, 5]]  # all possible predictors

# Step 1: select significance level => SL = 0.05 (5%)
# Step 2: fit the multiple linear regression model to our future optimal matrix of features X_opt & Y
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

# Step 3: remove predictor with the highest P-value
regressor_OLS.summary()
"""

num_rows, num_columns = X.shape
features_indexes = list(range(0, num_columns))
pvalues = [1]
while (max(pvalues) > 0.05):
    X_opt = X[:, features_indexes]  # all possible predictors
    regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
    regressor_OLS.summary()
    pvalues = regressor_OLS.pvalues
    max_pval_index = pvalues.tolist().index(max(pvalues))  # find out the highest P-value index
    features_indexes.pop(max_pval_index)
    print(features_indexes)
    print(max(pvalues))


