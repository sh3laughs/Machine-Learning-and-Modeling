# %%
# DSO106 - Machine Learning and Modeling
    # Lesson 6 - Introduction to Machine Learning
        # AKA: Machine Learning Lesson 1

# Page 3 - Supervised Machine Learning in Python Prep

# Import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# %%

# Import and preview data
housing = pd.read_csv('/Users/hannah/Library/CloudStorage/GoogleDrive-gracesnouveaux@gmail.com/My Drive/Bethel Tech/Data Science/DSO106 Machine Learning and Modeling/2:  Machine Learning – Lesson 1. Introduction to Machine Learning/realestate.csv')

housing

# %%
# 414 rows × 8 columns
    # Note: Dates are strange... funny that the column names have the X/ Y in 
        # name - convenient ;)

# Goal: accurately predict housing prices
    # IV (x-axis, continuous):
        # house age
        # distance to nearest MRT station
        # # of convenience stores
        # latitude
        # longitude
    # DV (y-axis, continuous): price
        # Note: Because this is continuous, I'm expecting us to use linear
        # regression, if we have regression in this excercise
    # H0: No IV's predict pricing
        # H1: At least one IV predicts price


# Wrangling

    # Create subsetted variable for IV
x = housing[['X2 house age', 'X3 distance to the nearest MRT station', 
    'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]

x

# %%
# 414 rows × 5 columns

    # Create subsetted variable for DV
y = housing[['Y house price of unit area']]

y

# %%
# 414 rows × 1 columns

    # Split data into train and test batches
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.4, 
    random_state = 101)

print(xTrain.shape, yTrain.shape)
print(xTest.shape, yTest.shape)

# %%
# (248, 5) (248, 1)
# (166, 5) (166, 1)


# Create and train a linear (I was right!) regression model
housingLm = LinearRegression()
housingLm.fit(xTrain, yTrain)

# %%
# LinearRegression
# LinearRegression()

# %%
# Page 4 - Supervised Machine Learning in Python

    # Create predictions variable, and preview predicted values
housingPredictions = housingLm.predict(xTest)

housingPredictions

# %%

# Interpret results

    # Graph test data against predicted for prices
plt.scatter(yTest, housingPredictions)

# %%
# Note: Data are not linear...

    # View accuracy score
print('Score:', housingLm.score(xTest, yTest))

# %%
# Score: 0.6442380845121716
    # Note: This doesn't seem noteworthy to me, but the lesson says it's great


    # Calculate mean absolute error (MAE)
metrics.mean_absolute_error(yTest, housingPredictions)

# %%
# 5.550201321415488
    # Note: This looks good to me!


    # Calculate mean squared error (MSE)
metrics.mean_squared_error(yTest, housingPredictions)

# %%
# 54.3757285449222
    # Note: Looks ok?


    # Calculate root mean squared error (RMSE)
np.sqrt(metrics.mean_squared_error(yTest, housingPredictions))

# %%
# 7.373990001683091
    # Note: Also looks great

# %%

# Page 7 - Cross Validation in Python

   # Create K Folds and view indeces for each group
housingKfolds = KFold(n_splits = 3, shuffle = True)

for train, test in housingKfolds.split(x, y):
    print('Train: %s, Test: %s' % (train, test))

# %%

# Accuracy score for K Folds model
print(cross_val_score(housingLm,x, y, cv = 3))

# %%
# [0.62051774 0.50393467 0.55970703]
    # Note: Doesn't seem super impressive...