# %%
# DSO106 - Machine Learning and Modeling
    # Lesson 7 - k-Means and k-Nearest Neighbors
        # AKA: Machine Learning Lesson 2

# Page 3 - k-Means in Python

    # From workshop - https://vimeo.com/528490801

# Import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# %%

# Import data
clients = pd.read_csv('/Users/hannah/Library/CloudStorage/GoogleDrive-gracesnouveaux@gmail.com/My Drive/Bethel Tech/Data Science/DSO106 Machine Learning and Modeling/2:  Machine Learning – Lesson 2. k-Means and k-Nearest Neighbors/clientSegmentation.csv')

clients

# %%
# 200 rows × 5 columns
    # Note: Nothing really stands out as needing attention - other than 
        # recoding Gender if we will use that


# Wrangling

    # Drop CustomerID and Gender
clients2 = clients.drop(['CustomerID', 'Gender'], axis = 1)

clients2

# %%
# 200 rows × 3 columns


# k-Means Analysis

    # Create k-Means model
clientsKm = KMeans(n_clusters = 2)

# %%

    # Fit the data to the model
clientsKm.fit(clients2)

# %%
# KMeans(n_clusters=2)


# Interpretation

    # Plot age and annual income
plt.figure(figsize = (10, 6))
plt.title('k Means')
plt.scatter(clients2.Age, clients2['Annual Income (k$)'], 
    c = clientsKm.labels_, cmap = 'viridis')

# %%
# Note: Age is the x axis, income the y axis
    # The plot does show a bit of a difference for < 40 y/o vs. > 40 y/o
    # For some reason the color in my scatterplots is reversed from the video...

    # Plot age and spending score
plt.figure(figsize = (10, 6))
plt.title('k Means')
plt.scatter(clients2.Age, clients2['Spending Score (1-100)'], 
    c = clientsKm.labels_, cmap = 'viridis')

# %%
# Note: Age is the x axis, spending score the y axis
    # Very little, if any overlap - the split is at spending < or > 40
    # My colors are reversed again...

        # Plot annual income and spending score
plt.figure(figsize = (10, 6))
plt.title('k Means')
plt.scatter(clients2['Annual Income (k$)'], clients2['Spending Score (1-100)'], 
    c = clientsKm.labels_, cmap = 'viridis')

# %%
# Note: Income is the x axis, spending the y axis
    # Here there's an overlap between 40-60 spending score


    # Add labels to data
clients2['Category'] = clientsKm.labels_

clients2

# %%
# 200 rows × 4 columns

        # Investigate means by category
clients2.groupby('Category')['Age'].mean()

# %%
# Category
# 0    46.165217
# 1    28.952941
# Name: Age, dtype: float64
    # Note: This is saying that the 0 category is older (~46 y/o) and 1 is 
        # younger (~29 y/o)

    # Investigate means by income
clients2.groupby('Category')['Annual Income (k$)'].mean()

# %%
# Category
# 0    59.365217
# 1    62.176471
# Name: Annual Income (k$), dtype: float64
    # Note: This is saying that the 0 category has a lower annual income 
        # (~$59k) and 1 has a higher income (~$62k)

    # Investigate means by spending
clients2.groupby('Category')['Spending Score (1-100)'].mean()

# %%
# Category
# 0    32.886957
# 1    73.623529
# Name: Spending Score (1-100), dtype: float64
    # Note: This is saying that the 0 category has a lower spending score (~33)
        # and 1 has a higher spending score (~74)... I'm not sure if the score
        # means they spend more or less the higher it goes... 
    
    # View category distribution
clients2.Category.value_counts()

# %%
# 0    115
# 1     85
# Name: Category, dtype: int64
    # Note: There are more people in category 0 than 1

# %%

   # From lesson

# Import data
iris = sns.load_dataset('iris')

iris

# %%
# 150 rows × 5 columns
    # Note: Only wrangling I see is to remove or recode the species column, per 
        # being string


# Wrangling

    # Drop species
iris2 = iris.drop('species', axis = 1)

iris2

# %%
# 150 rows × 4 columns


# k-Means Analysis

    # Create k-Means model
irisKm = KMeans(n_clusters = 2)

# %%

    # Fit the data to the model
irisKm.fit(iris2)

# %%
# KMeans(n_clusters=2)


# Interpretation

    # Plot petal length (x) and width (y)
# Interpretation

    # Plot age and annual income
plt.figure(figsize = (10, 6))
plt.title('k Means')
plt.scatter(iris2.petal_length, iris2.petal_width, c = irisKm.labels_, 
    cmap = 'viridis')

# %%
# Note: There is very little, if any overlap, the division seems to be around
    # a length of 3 and width of 1


    # Add labels to data
iris2['Category'] = irisKm.labels_

iris2

# %%
# 150 rows × 5 columns


    # Find center points
irisKm.cluster_centers_

# %%
# array([[6.30103093, 2.88659794, 4.95876289, 1.69587629],
#        [5.00566038, 3.36981132, 1.56037736, 0.29056604]])
    # Note: I'm not totally clear on what these are, and the lesson doesn't tell
        # us... I can tell that it's 2 (per 2 clusters) lists of 4 (per 4
        # columns), but not more than that... ie: is a center point different
        # from the mean?


    # Find inertia – total distance from every point to the cluster center
irisKm.inertia_

# %%
# 152.3479517603579
    # Note: Also not sure what value this single # has for me, and lesson 
        # doesn't help with that

# %%

# Page 5 - kNN Setup in Python

    # From workshop - https://vimeo.com/528059230

# Preview data (same as from page 3 workshop)
clients

# %%
# 200 rows × 5 columns
    # Note: Gender will likely be DV, per being categorical

# Goal: Predict gender based on age, annual income, and spending score
    # IV's (x axis, continuous): age, annual income, spending score
    # DV (y axis, categorical): gender


# Wrangling

    # Drop CustomerID and Gender

clients3 = clients.drop(['CustomerID', 'Gender'], axis = 1)

clients3

# %%
# 200 rows × 3 columns


    # Scale the data
clientsScaler = StandardScaler()
clientsScaler.fit(clients3)
clientsScaling = clientsScaler.transform(clients3)
clientsScaled = pd.DataFrame(clientsScaling)

clientsScaled

# %%
# 200 rows × 3 columns

        # Rename columns
clientsScaled.rename(columns = {0: 'age', 1: 'annualIncome', 2: 
    'spendingScore'}, inplace = True)

clientsScaled

# %%
# 200 rows × 3 columns
    # Note: Success!


    # Define x variable
x = clientsScaled

x

# %%
# 200 rows × 3 columns

    # Define y variable by subsetting
y = clients.Gender

y

# %%
# Name: Gender, Length: 200, dtype: object
    # Note: This is not a dataframe, will try an alternate method

y = clients['Gender']

y

# %%
# Note: Same results, so this data type must not matter...


    # Create train / test split of data
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3, 
    random_state = 101)

# %%


# kNN Analysis

    # Specify quantity of neighbors
clientErrorRate = []

for client in range(1, 40):
    clientsKnn = KNeighborsClassifier(n_neighbors = client)
    clientsKnn.fit(xTrain, yTrain)
    clientsPrediction = clientsKnn.predict(xTest)
    clientErrorRate.append(np.mean(clientsPrediction != yTest))

# %%

# Interpret

    # Plot
plt.figure(figsize = (10, 6))
plt.plot(range(1, 40), clientErrorRate, color = 'blue', linestyle = 'dashed',
    marker = 'o', markerfacecolor = 'red', markersize = 10)
plt.title('Error Rate vs. k Value')
plt.xlabel('k')
plt.ylabel('Error Rate')

# %%
# Note: The lowest data point appears to be around 17 on the y axis (k value),
    # so we want to look at 17 k-nearest neighbors


    # Create new model based off of 17 k-nearest neighbors
clientsKnn2 = KNeighborsClassifier(n_neighbors = 17)
clientsKnn2.fit(xTrain, yTrain)
clientsPredictions2 = clientsKnn2.predict(xTest)

# %%

        # View confusion matrix for new model
print(confusion_matrix(yTest, clientsPredictions2))

# %%
# [[32  3]
#  [18  7]]
    # Note: 32 = TP, 3 = FP, 18 = FN, 7 = TN
        # We were trying to predict gender, but I'm not sure (and video 
        # doesn't clarify) how to know which gender is positive, and thus which
        # is negative
        
    
        # View classification report
print(classification_report(yTest, clientsPredictions2))

# %%
#               precision    recall  f1-score   support

#       Female       0.64      0.91      0.75        35
#         Male       0.70      0.28      0.40        25

#     accuracy                           0.65        60
#    macro avg       0.67      0.60      0.58        60
# weighted avg       0.67      0.65      0.61        60
    # Note: This model is a little better at predicting males than females, 
        # based on their age, income, and spending score
        # Overall this model predicts gender with 67% accuracy

# %%

    # From lesson

# Preview data (same as from page 3 lesson)
iris

# %%
# 150 rows × 5 columns
    # Note: Species will likely be DV, per being categorical

# Goal: Predict species of iris


# Wrangling

    # Scale data and drop DV
irisScaler = StandardScaler()
irisScaler.fit(iris.drop('species', axis = 1))
irisScaling = irisScaler.transform(iris.drop('species', axis = 1))
irisScaled = pd.DataFrame(irisScaling, columns = iris.columns[:-1])

irisScaled

# %%
# 150 rows × 4 columns
    # Note: I like how this consolidated the wrangling steps into less lines


    # Create x variable for scaled IV's
x = irisScaled

x

# %%
# 150 rows × 4 columns

    # Create y variable for DV
y = iris['species']

y

# %%
# 0         setosa
# 1         setosa
# 2         setosa
# 3         setosa
# 4         setosa
#          ...    
# 145    virginica
# 146    virginica
# 147    virginica
# 148    virginica
# 149    virginica
# Name: species, Length: 150, dtype: object


    # Create train and test data splits
xTrain2, xTest2, yTrain2, yTest2 = train_test_split(x, y, test_size = 0.3, 
    random_state = 101)

# %%

# Page 6 - KNN in Python

# Build and run initial model

    # Create initial model
irisKnn = KNeighborsClassifier(n_neighbors = 1)

# %%

    # Train / fit initial model
irisKnn.fit(xTrain2, yTrain2)

# %%
# KNeighborsClassifier(n_neighbors=1)

    # Run initial model
irisPredictions = irisKnn.predict(xTest2)

# %%

# Interpret

    # Confusion matrix
print(confusion_matrix(yTest2, irisPredictions))

# %%
# [[13  0  0]
#  [ 0 19  1]
#  [ 0  1 11]]
    # Note: Because species has 3 values, apparently, we have a 3x3 grid... 
        # though I'm not sure then how this maps to the TP, FP, FN, TN idea...
        # that said, the 13, 9, 11 are the accurate predictions and 0, 0, 0, 1,
        # 0, 1 are inaccurate predictions, so it seems like this model is 
        # either highly accurate or overfitted


    # Classification report for accuracy scores
print(classification_report(yTest2, irisPredictions))

# %%
#               precision    recall  f1-score   support

#       setosa       1.00      1.00      1.00        13
#   versicolor       0.95      0.95      0.95        20
#    virginica       0.92      0.92      0.92        12

#     accuracy                           0.96        45
#    macro avg       0.96      0.96      0.96        45
# weighted avg       0.96      0.96      0.96        45
    # Note: TBH, this seems overfitted – but as it reads, the model has an 
        # overall accuracy of 96% in predicting iris species
        # It's 100% accurate in predicting the setosa species, 95% for 
        # versicolor, and 92% for virginica


# Improve model (pretending these numbers weren't crazy high already, I guess)

    # Create error variable
irisErrorRate = []

for iris in range(1, 40):
    irisKnn2 = KNeighborsClassifier(n_neighbors = iris)
    irisKnn2.fit(xTrain2, yTrain2)
    irisPredictions2 = irisKnn2.predict(xTest2)
    irisErrorRate.append(np.mean(irisPredictions2 != yTest2))

# %%

    # Plot errors
plt.figure(figsize = (10, 6))
plt.plot(range(1, 40), irisErrorRate, color = 'blue', linestyle = 'dashed', 
    marker = 'o', markerfacecolor = 'red', markersize = 10)
plt.title('Error Rate vs. k Value')
plt.xlabel('k')
plt.ylabel('Error Rate')

# %%
# Note: 7, 8, 9, 11, 12 are all about 0 (0 errors), so we could pick any of 
    # those, or their average, for new model – lesson picks 8

    # Build and run new model
irisKnn3 = KNeighborsClassifier(n_neighbors = 8)
irisKnn3.fit(xTrain2, yTrain2)
irisPredictions3 = irisKnn3.predict(xTest2)

# %%

    # Confustion matrix
print(confusion_matrix(yTest2, irisPredictions3))

# %%
# [[13  0  0]
#  [ 0 20  0]
#  [ 0  0 12]]
    # Note: 100% accurate!


    # Classification report
print(classification_report(yTest2, irisPredictions3))

# %%
#               precision    recall  f1-score   support

#       setosa       1.00      1.00      1.00        13
#   versicolor       1.00      1.00      1.00        20
#    virginica       1.00      1.00      1.00        12

#     accuracy                           1.00        45
#    macro avg       1.00      1.00      1.00        45
# weighted avg       1.00      1.00      1.00        45
    # Note: 100% accuracy confirmed 