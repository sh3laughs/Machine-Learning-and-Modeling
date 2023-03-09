# %%
# DSO106 - Machine Learning and Modeling
    # Lesson 6 - Introduction to Machine Learning
        # AKA: Machine Learning Lesson 1
    # Page 9 - Machine Learning Basics Hands-On

# Requirements: Now that you've learned your first machine learning algorithm, 
    # it's time to put that knowledge to work. In this Hands-On exercise you 
    # will create a project which will require you to take data, clean it so 
    # that it's usable, and finally create a linear model to predict unknown 
    # data. This Hands-On project should be completed using the browser for 
    # downloading data and Python for plotting and modeling the data.
# You should leverage what you have learned about machine learning and data 
    # modeling. Import the diamonds dataset from seaborn using this code:
        # import seaborn as sns
        # from sklearn.utils import shuffle
        # Diamonds = shuffle(sns.load_dataset('diamonds'))
        # If seaborn isn't working for you, click here to download the data.
    # And use the following variables to predict the price of diamonds:
        # carat
        # cut
        # color
        # clarity
# You will need to utilize the train_test_split() method as well as 
    # LinearRegression() to train and test your algorithm. Then, leverage your 
    # knowledge of cross-validation and Python programming to cross-validate 
    # the work you did. Note the variation in model accuracy once you have 
    # cross-validated the model using 5 iterations.

# Goal: Determine whether carat, cut, color, and clarity predict the price of
        # diamonds
    # IV's (x axis, continuous):
        # carat
        # cut
        # color
        # clarity
    # DV (y axis, continuous): price


# Import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# %%
diamonds = shuffle(sns.load_dataset('diamonds'))

diamonds

# %%
# 53940 rows × 10 columns
    # Note: That's a lotta data!
        # Cut, color and clarity will need to be recoded, as they are not 
        # currently numeric / continous

# Wrangling

    # Recode non-numeric IV's (the long way)

        # Confirm unique values for cut
diamonds.cut.unique()

# %%
# ['Premium', 'Good', 'Ideal', 'Very Good', 'Fair']
# Categories (5, object): ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair']

        # Confirm unique values for color
diamonds.color.unique()

# %%
# ['H', 'D', 'G', 'J', 'E', 'F', 'I']
# Categories (7, object): ['D', 'E', 'F', 'G', 'H', 'I', 'J'

        # Confirm unique values for clarity
diamonds.clarity.unique()

# %%
# ['SI1', 'VS1', 'SI2', 'VVS1', 'VVS2', 'VS2', 'IF', 'I1']
# Categories (8, object): ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 
    # 'I1']

        # Recode cut
def cutRecode (cutType):
        if cutType == 'Fair':
            return 0
        if cutType == 'Good':
            return 1
        if cutType == 'Very Good':
            return 2
        if cutType == 'Ideal':
            return 3
        if cutType == 'Premium':
            return 4

diamonds['cutR'] = diamonds.cut.apply(cutRecode)

diamonds

# %%
# 53940 rows × 11 columns

        # Recode color
def colorRecode (colorCode):
        if colorCode == 'D':
            return 0
        if colorCode == 'E':
            return 1
        if colorCode == 'F':
            return 2
        if colorCode == 'G':
            return 3
        if colorCode == 'H':
            return 4
        if colorCode == 'I':
            return 5
        if colorCode == 'J':
            return 6

diamonds['colorR'] = diamonds.color.apply(colorRecode)

diamonds

# %%
# 53940 rows × 12 columns

        # Recode clarity
def clarityRecode (clarityCode):
        if clarityCode == 'IF':
            return 0
        if clarityCode == 'VVS1':
            return 1
        if clarityCode == 'VVS2':
            return 2
        if clarityCode == 'VS1':
            return 3
        if clarityCode == 'VS2':
            return 4
        if clarityCode == 'SI1':
            return 5
        if clarityCode == 'SI2':
            return 6
        if clarityCode == 'I1':
            return 7

diamonds['clarityR'] = diamonds.clarity.apply(clarityRecode)

diamonds

# %%
# 53940 rows × 13 columns

        # Confirm data types for all columns
diamonds.info()

# %%
# Note: Recoded columns are categorical

        # Update data types for recoded variables
diamonds2 = diamonds.astype({'cutR': 'int', 'colorR': 'int', 
    'clarityR': 'int'})

diamonds2.info()

# %%
# Note: Success! 

    # Create subsetted variable for IV
x = diamonds2[['carat', 'cutR', 'colorR', 'clarityR']]

x

# %%
# 53940 rows × 4 columns

    # Create subsetted variable for DV
y = diamonds2[['price']]

y

# %%
# 53940 rows × 1 columns

    # Split data into train and test batches and print the shape of these   
        # groups – though lesson content had used a 60/40 split, I'm choosing 
        # to use a 70/30
        # Also, because there are over 50k rows, I've increased the random seed
        # to 1,001 from 101 in the lesson content
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3, 
    random_state = 1001)

print(xTrain.shape, yTrain.shape)
print(xTest.shape, yTest.shape)

# %%
# (37758, 5) (37758, 1)
# (16182, 5) (16182, 1)


# Create and train a linear regression model
diamondsLm = LinearRegression()
diamondsLm.fit(xTrain, yTrain)

# %%
# LinearRegression()


    # Create predictions variable, and preview predicted values
diamondsPredictions = diamondsLm.predict(xTest)

diamondsPredictions

# %%
# array([[12540.87197948],
#        [ 3883.51120021],
#        [ 5476.03803465],
#        ...,
#        [ 1190.94758384],
#        [ 1538.25798195],
#        [ 5752.58617621]])


# Interpret results

    # Graph test data against predicted for prices
plt.scatter(yTest, diamondsPredictions)

# %%
# Note: This is not perfectly linear (which is a good thing, actually), but
    # close enough to indicate this model will be helpful


    # View accuracy score
print('Score:', diamondsLm.score(xTest, yTest))

# %%
# Score: 0.905873148310784
    # Note: Pretty good!! In fact maybe high enough to indicate overfitting 
        # since there was a lot of data!


    # Calculate mean absolute error (MAE)
metrics.mean_absolute_error(yTest, diamondsPredictions)

# %%
# 852.8877152657495
    # Note: This doesn't seem great, but may actually be ok given the quantity 
        # of data its based on (not sure whether that is relevant in analyzing 
        # these calculations for error means)


    # Calculate mean squared error (MSE)
metrics.mean_squared_error(yTest, diamondsPredictions)

# %%
# 1488786.1828466796
    # Note: This seems too high to indicate value in the model


    # Calculate root mean squared error (RMSE)
np.sqrt(metrics.mean_squared_error(yTest, diamondsPredictions))

# %%
# 1220.158261393447
    # Note: Also doesn't seem great, let's see how K Folds cross-validation
        # impacts things!


   # Create K Folds and view indeces for each group
diamondsKfolds = KFold(n_splits = 5, shuffle = True)

for train, test in diamondsKfolds.split(x, y):
    print('Train: %s, Test: %s' % (train, test))

# %%
# Train: [    0     1     2 ... 53936 53937 53938], 
    # Test: [    4    11    16 ... 53927 53933 53939]
# Train: [    0     1     2 ... 53936 53937 53939], 
    # Test: [    3     7    12 ... 53924 53935 53938]
# Train: [    1     2     3 ... 53937 53938 53939], 
    # Test: [    0     6     8 ... 53929 53930 53932]
# Train: [    0     1     3 ... 53936 53938 53939], 
    # Test: [    2    15    18 ... 53913 53931 53937]
# Train: [    0     2     3 ... 53937 53938 53939], 
    # Test: [    1     5     9 ... 53928 53934 53936]
        # Note: A lotta numbers ;)


# Accuracy score for K Folds model
print(cross_val_score(diamondsLm,x, y, cv = 5))

# %%
# [0.90314191 0.89839171 0.90339461 0.90739632 0.90550645]
    # Note: These are helpful, most likely in confirming the model is overfitted
        # I'll try another version with less of the data to see if that makes
        # the predictions more realistic... as well as using a 60/40 train/test
        # split...


# Subset data to try to prevent overfitting
diamonds3 = diamonds2[0:25000]

diamonds3

# %%
# 25000 rows × 13 columns

    # Create sub-subsetted variable for IV
x3 = diamonds3[['carat', 'cutR', 'colorR', 'clarityR']]

x3

# %%
# 25000 rows × 4 columns

    # Create subsetted variable for DV
y3 = diamonds3[['price']]

y3

# %%
# 25000 rows × 1 columns

    # Split data into train and test batches – using 60/40
xTrain3, xTest3, yTrain3, yTest3 = train_test_split(x3, y3, test_size = 0.4, 
    random_state = 1001)

print(xTrain3.shape, yTrain3.shape)
print(xTest3.shape, yTest3.shape)

# %%
# (15000, 4) (15000, 1)
# (10000, 4) (10000, 1)


# Create and train a linear regression model
diamondsLm3 = LinearRegression()
diamondsLm3.fit(xTrain3, yTrain3)

# %%
# LinearRegression()


    # Create predictions variable, and preview predicted values
diamondsPredictions3 = diamondsLm3.predict(xTest3)

diamondsPredictions3

# %%
# array([[10745.02232805],
#        [  887.68024312],
#        [ 1478.13815617],
#        ...,
#        [ 3096.72769612],
#        [ 2514.43550545],
#        [ 2325.40050811]])


# Interpret results

    # Graph test data against predicted for prices
plt.scatter(yTest3, diamondsPredictions3)

# %%
# Note: This is similar to the original model's predictions linearity - it is
    # a little less linear, which is helpful...


    # View accuracy score
print('Score:', diamondsLm3.score(xTest3, yTest3))

# %%
# Score: 0.9010764561422081
    # Note: Unfortunately this doesn't look any more realistic than the 
        # original...


    # Calculate mean absolute error (MAE)
metrics.mean_absolute_error(yTest3, diamondsPredictions3)

# %%
# 852.1032787330433
    # Note: Only a tiny difference from the original model, but will keep 
        # analyzing...


    # Calculate mean squared error (MSE)
metrics.mean_squared_error(yTest3, diamondsPredictions3)

# %%
# 1545878.1904211256
    # Note: Higher than the original, which makes sense...


    # Calculate root mean squared error (RMSE)
np.sqrt(metrics.mean_squared_error(yTest3, diamondsPredictions3))

# %%
# 1243.3334992756872
    # Note: Not enough of a difference from the original to give me hope


   # Create K Folds and view indeces for each group
diamondsKfolds3 = KFold(n_splits = 5, shuffle = True)

for train3, test3 in diamondsKfolds3.split(x, y):
    print('Train: %s, Test: %s' % (train3, test3))

# %%
# Train: [    0     1     2 ... 53937 53938 53939], 
    # Test: [    3     7    13 ... 53923 53935 53936]
# Train: [    1     2     3 ... 53935 53936 53938], 
    # Test: [    0     6    12 ... 53920 53937 53939]
# Train: [    0     2     3 ... 53937 53938 53939], 
    # Test: [    1     4     9 ... 53928 53932 53933]
# Train: [    0     1     2 ... 53937 53938 53939], 
    # Test: [   14    18    22 ... 53926 53930 53934]
# Train: [    0     1     3 ... 53936 53937 53939], 
    # Test: [    2     5     8 ... 53929 53931 53938]


# Accuracy score for K Folds model
print(cross_val_score(diamondsLm3,x3, y3, cv = 5))

# %%
# [0.90781547 0.90190289 0.89950762 0.89837367 0.90198789]
    # Note: Unfortunately this is so minimally different, it is just as 
        # unreliable as the original model

# %%
# Summary: This model is able to predict the price of diamonds with about 90%
    # accuracy, which makes it a bit unreliable. To make this model more useful
    # the overfitting issue would need to be addressed (which we haven't 
    # learned how to do yet)