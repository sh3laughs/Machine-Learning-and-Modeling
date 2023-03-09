# %%
# DSO106 - Machine Learning and Modeling
    # Lesson 8 - Decision Trees and Random Forests
        # AKA: Machine Learning Lesson 3

# Page 4 - Decision Trees in Python

    # From Workshop - https://vimeo.com/528497271

# Import packages
import matplotlib.pyplot as plt
import pandas as pd
import quandl
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# %%

# Import data
telOriginal = pd.read_csv('/Users/hannah/Library/CloudStorage/GoogleDrive-gracesnouveaux@gmail.com/My Drive/Bethel Tech/Data Science/DSO106 Machine Learning and Modeling/2:  Machine Learning – Lesson 3. Decision Trees and Random Forests/telcomChurn.csv')

telOriginal

# %%
# 7043 rows × 21 columns
    # Note: I can't see all columns in preview

    # Update display to include all columns
pd.options.display.max_columns = 50

telOriginal

# %%

# Wrangling

    # Drop customerID
tel1 = telOriginal.drop('customerID', axis = 1)

tel1

# %%
# 7043 rows × 20 columns


    # Recode string variables

        # Recode gender
def genderFunction (mf):
    if mf == 'Male':
        return 0
    if mf == 'Female':
        return 1

tel1['genderR'] = tel1.gender.apply(genderFunction)

tel1

# %%
# 7043 rows × 21 columns

        # Find unique values for additional variables

            # Partner
tel1.Partner.value_counts()

# %%
# No     3641
# Yes    3402
# Name: Partner, dtype: int64

            # Dependents
tel1.Dependents.value_counts()

# %%
# No     4933
# Yes    2110
# Name: Dependents, dtype: int64

            # PhoneService
tel1.PhoneService.value_counts()

# %%
# Yes    6361
# No      682
# Name: PhoneService, dtype: int64

            # MultipleLines
tel1.MultipleLines.value_counts()

# %%
# No                  3390
# Yes                 2971
# No phone service     682
# Name: MultipleLines, dtype: int64

            # InternetService
tel1.InternetService.value_counts()

# %%
# Fiber optic    3096
# DSL            2421
# No             1526
# Name: InternetService, dtype: int64

            # OnlineSecurity
tel1.OnlineSecurity.value_counts()

# %%
# No                     3498
# Yes                    2019
# No internet service    1526
# Name: OnlineSecurity, dtype: int64

            # OnlineBackup
tel1.OnlineBackup.value_counts()

# %%
# No                     3088
# Yes                    2429
# No internet service    1526
# Name: OnlineBackup, dtype: int64

            # DeviceProtection
tel1.DeviceProtection.value_counts()


# %%
# No                     3095
# Yes                    2422
# No internet service    1526
# Name: DeviceProtection, dtype: int64

            # TechSupport
tel1.TechSupport.value_counts()

# %%
# No                     3473
# Yes                    2044
# No internet service    1526
# Name: TechSupport, dtype: int64

            # StreamingTV
tel1.StreamingTV.value_counts()

# %%
# No                     2810
# Yes                    2707
# No internet service    1526
# Name: StreamingTV, dtype: int64

            # StreamingMovies
tel1.StreamingMovies.value_counts()

# %%
# No                     2785
# Yes                    2732
# No internet service    1526
# Name: StreamingMovies, dtype: int64

            # Contract
tel1.Contract.value_counts()

# %%
# Month-to-month    3875
# Two year          1695
# One year          1473
# Name: Contract, dtype: int64

            # PaperlessBilling
tel1.PaperlessBilling.value_counts()

# %%
# Yes    4171
# No     2872
# Name: PaperlessBilling, dtype: int64

            # PaymentMethod
tel1.PaymentMethod.value_counts()

# %%
# Electronic check             2365
# Mailed check                 1612
# Bank transfer (automatic)    1544
# Credit card (automatic)      1522
# Name: PaymentMethod, dtype: int64

            # Churn
tel1.Churn.value_counts()

# %%
# No     5174
# Yes    1869
# Name: Churn, dtype: int64

        # Recode 5x variables with yes / no values (only)
def yesNoFunction (yn):
    if yn == 'No':
        return 0
    if yn == 'Yes':
        return 1

tel1['partnerR'] = tel1.Partner.apply(yesNoFunction)
tel1['dependentsR'] = tel1.Dependents.apply(yesNoFunction)
tel1['phoneServiceR'] = tel1.PhoneService.apply(yesNoFunction)
tel1['paperlessBillingR'] = tel1.PaperlessBilling.apply(yesNoFunction)
tel1['churnR'] = tel1.Churn.apply(yesNoFunction)

tel1

# %%
# 7043 rows × 26 columns
    # Note: Learned later in workshop that Churn actually didn't need to be 
        # recoded, per being our categorical DV

        # Recode MultipleLines
def multiLineFunction (line):
    if line == 'No':
        return 0
    if line == 'Yes':
        return 1
    if line == 'No phone service':
        return 2

tel1['multipleLinesR'] = tel1.MultipleLines.apply(multiLineFunction)

tel1

# %%
# 7043 rows × 27 columns

        # Recode InternetService
def intServFunction (line):
    if line == 'No':
        return 0
    if line == 'Fiber optic':
        return 1
    if line == 'DSL':
        return 2

tel1['internetServiceR'] = tel1.InternetService.apply(intServFunction)

tel1

# %%
# 7043 rows × 28 columns

        # Recode 6x variables with Yes / No / No internet service
def noIntServFunction (line):
    if line == 'No':
        return 0
    if line == 'Yes':
        return 1
    if line == 'No internet service':
        return 2

tel1['onlineSecurityR'] = tel1.OnlineSecurity.apply(noIntServFunction)
tel1['onlineBackupR'] = tel1.OnlineBackup.apply(noIntServFunction)
tel1['deviceProtectionR'] = tel1.DeviceProtection.apply(noIntServFunction)
tel1['techSupportR'] = tel1.TechSupport.apply(noIntServFunction)
tel1['streamingTvR'] = tel1.StreamingTV.apply(noIntServFunction)
tel1['streamingMoviesR'] = tel1.StreamingMovies.apply(noIntServFunction)

tel1


# %%
# 7043 rows × 34 columns

        # Recode Contract
def contractFunction (line):
    if line == 'Month-to-month':
        return 0
    if line == 'One year':
        return 1
    if line == 'Two year':
        return 2

tel1['contractR'] = tel1.Contract.apply(contractFunction)

tel1

# %%
# 7043 rows × 35 columns

        # Recode PaymentMethod
def pymtMethFunction (line):
    if line == 'Mailed check':
        return 0
    if line == 'Electronic check':
        return 1
    if line == 'Bank transfer (automatic)':
        return 2
    if line == 'Credit card (automatic)':
        return 3

tel1['paymentMethodR'] = tel1.PaymentMethod.apply(pymtMethFunction)

tel1

# %%
# 7043 rows × 36 columns


    # Confirm data types for all variables
tel1.info()

# %%
# 0   gender             7043 non-null   object - recoded
#  1   SeniorCitizen      7043 non-null   int64  
#  2   Partner            7043 non-null   object - recoded
#  3   Dependents         7043 non-null   object - recoded
#  4   tenure             7043 non-null   int64  
#  5   PhoneService       7043 non-null   object - recoded
#  6   MultipleLines      7043 non-null   object - recoded
#  7   InternetService    7043 non-null   object - recoded
#  8   OnlineSecurity     7043 non-null   object - recoded
#  9   OnlineBackup       7043 non-null   object - recoded
#  10  DeviceProtection   7043 non-null   object - recoded
#  11  TechSupport        7043 non-null   object - recoded
#  12  StreamingTV        7043 non-null   object - recoded
#  13  StreamingMovies    7043 non-null   object - recoded
#  14  Contract           7043 non-null   object - recoded
#  15  PaperlessBilling   7043 non-null   object - recoded
#  16  PaymentMethod      7043 non-null   object - recoded
#  17  MonthlyCharges     7043 non-null   float64
#  18  TotalCharges       7043 non-null   object - NEED TO UPDATE
#  19  Churn              7043 non-null   object - recoded
#  20  genderR            7043 non-null   int64  
#  21  partnerR           7043 non-null   int64  
#  22  dependentsR        7043 non-null   int64  
#  23  phoneServiceR      7043 non-null   int64  
#  24  paperlessBillingR  7043 non-null   int64  
#  25  churnR             7043 non-null   int64  
#  26  multipleLinesR     7043 non-null   int64  
#  27  internetServiceR   7043 non-null   int64  
#  28  onlineSecurityR    7043 non-null   int64  
#  29  onlineBackupR      7043 non-null   int64  
#  30  deviceProtectionR  7043 non-null   int64  
#  31  techSupportR       7043 non-null   int64  
#  32  streamingTvR       7043 non-null   int64  
#  33  streamingMoviesR   7043 non-null   int64  
#  34  contractR          7043 non-null   int64  
#  35  paymentMethodR     7043 non-null   int64  
# dtypes: float64(1), int64(18), object(17)
# memory usage: 1.9+ MB

        # Convert TotalCharges to an integer
tel1['TotalCharges'] = pd.to_numeric(tel1.TotalCharges, errors = 'coerce')

tel1.info()

# %%
# Note: Success!


    # Drop missing values
tel1.dropna(inplace = True)

tel1

# %%
# 7032 rows × 36 columns
    # Note: 11 rows removed


    # Define x variable for IV's

        # New variable for non-recoded IV's
x1 = tel1[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']]

x1 

# %%
# 7032 rows × 4 columns

        # New variable for first batch of recoded variables before churnR
x2Columns1 = range(20, 24)
x2Columns2 = range(26, 35)
x2Columns = list(x2Columns1) + list(x2Columns2)
x2 = tel1.iloc[:, x2Columns]

x2

# %%
# 7032 rows × 13 columns

        # Merge into single x variable
x = pd.concat([x1, x2], axis = 1)

x

# %%
# 7032 rows × 17 columns

    # Define y variable for DV
y = pd.DataFrame(tel1.Churn)

y

# %%
# 7032 rows × 1 column


    # Create train / test split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3)

# %%


# Create initial decision tree model
telDecTree = DecisionTreeClassifier()

# %%

    # Train the model
telDecTree.fit(xTrain, yTrain)

# %%
# DecisionTreeClassifier()

    # Test the model
telPredictions = telDecTree.predict(xTest)

# %%


# Interpret

    # View confusion matrix
print(confusion_matrix(yTest, telPredictions))

# %%
# [[1275  282]
#  [ 291  262]]
    # Note: 1275 and 262 = accurate predictions / 282 and 291 = inaccurate
        # predictions... not too shabby
    

    # View classification report
print(classification_report(yTest, telPredictions))

# %%
#               precision    recall  f1-score   support
#           No       0.81      0.82      0.82      1557
#          Yes       0.48      0.47      0.48       553

#     accuracy                           0.73      2110
#    macro avg       0.65      0.65      0.65      2110
# weighted avg       0.73      0.73      0.73      2110
    # Note: This model is better at predicting customers who will stick around
        # than customers who will leave.. which actually seems less useful
        # than the reverse


# Create initial random forest model
telRandForest = RandomForestClassifier()

# %%

    # Train model
telRandForest.fit(xTrain, yTrain)

# %%

    # Test model
telRFpredictions = telRandForest.predict(xTest)

# %%


# Interpret

    # View accuracy score
print(accuracy_score(yTest, telRFpredictions))

# %%
# 0.7876777251184834
    # Note: pretty good!


    # View classification report
print(classification_report(yTest, telRFpredictions))

# %%
#               precision    recall  f1-score   support
#           No       0.83      0.90      0.86      1557
#          Yes       0.62      0.48      0.54       553

#     accuracy                           0.79      2110
#    macro avg       0.73      0.69      0.70      2110
# weighted avg       0.78      0.79      0.78      2110
    # Note: This model is also better at predicting customers who will stick 
        # around than customers who will leave..

# %%

    # From lesson

# Import data
iris = sns.load_dataset('iris')

iris

# %%
# 150 rows × 5 columns
    # Note: Doesn't look like anything needs to be recoded, species is DV

# Wrangling

    # Define x variable for IV's
xIris = iris.drop('species', axis = 1)

xIris

# %%
# 150 rows × 4 columns

    # Define y variable for DV
yIris = iris['species']

yIris

# %%
#150 rows × 1 columns

    # Create train / test split
xTrain2, xTest2, yTrain2, yTest2 = train_test_split(xIris, yIris, 
    test_size = 0.3, random_state = 76)

# %%
# Create initial decision tree model
irisDecTree = DecisionTreeClassifier(random_state = 76)

# %%

    # Train the model
irisDecTree.fit(xTrain2, yTrain2)

# %%
# DecisionTreeClassifier(random_state=76)


    # Run the model
irisPredictions = irisDecTree.predict(xTest2)

# %%

# Interpret

    # Confusion Matrix
print(confusion_matrix(yTest2, irisPredictions))

# %%
# [[19  0  0]
#  [ 0 10  3]
#  [ 0  2 11]]
    # 19 = accurately predicted setosa
    # 10 = accurately predicted versicolor
    # 11 = accurately predicted virginica
    # 3 and 2 were the only inaccurate predictions, which is very low (too
        # good to be true?)


    # Classification report
print(classification_report(yTest2, irisPredictions))

# %%
#               precision    recall  f1-score   support

#       setosa       1.00      1.00      1.00        19
#   versicolor       0.83      0.77      0.80        13
#    virginica       0.79      0.85      0.81        13

#     accuracy                           0.89        45
#    macro avg       0.87      0.87      0.87        45
# weighted avg       0.89      0.89      0.89        45
    # Note: The model is 100% accurate at predicting setosa (too good to be 
        # true?), about 83% accurate at predicting versicolor, and 79% for
        # virginica; it's about 89% accurate overall

# %%

# Page 5 - Random Forests in Python

# Wrangling

    # Would create the same x & y variables as prior page, so we can use those

    # Likewise, will re-use same train & test split variables


# Create initial random forest model
irisRandFor = RandomForestClassifier(n_estimators = 500, random_state = 76)

# %%

    # Train model
irisRandFor.fit(xTrain2, yTrain2)

# %%
# RandomForestClassifier(n_estimators=500, random_state=76)


    # Run model
irisRandForPredictions = irisRandFor.predict(xTest2)

# %%

# Interpret

    # Confusion matrix

        # Create labels
irisLabels = ['setosa', 'versicolor', 'virginica']

irisLabels

# %%
# ['setosa', 'versicolor', 'virginica']

        # Save labeled matrix as new variable
irisCm = confusion_matrix(yTest2, irisRandForPredictions, labels = irisLabels)

irisCm

# %%
# array([[19,  0,  0],
#        [ 0, 11,  2],
#        [ 0,  0, 13]])

        # Create dataframe as labeled matrix
irisCmLabeled = pd.DataFrame(irisCm, index = irisLabels, columns = irisLabels)

irisCmLabeled

# %%
# 	        setosa	versicolor	virginica
# setosa	    19	    0	        0
# versicolor	0	    11	        2
# virginica	    0	    0	        13
    # Note: This predicted one more versicolor and two more virginica 
        # accurately (as compared with the single decision tree model), making
        # it 100% accurate for both setosa and virginica - though twice it
        # predicted a versicolor as a virginica, which is why the versicolor
        # is not also 100%


    # Classification report
print(classification_report(yTest2, irisRandForPredictions))

# %%
#               precision    recall  f1-score   support

#       setosa       1.00      1.00      1.00        19
#   versicolor       1.00      0.85      0.92        13
#    virginica       0.87      1.00      0.93        13

#     accuracy                           0.96        45
#    macro avg       0.96      0.95      0.95        45
# weighted avg       0.96      0.96      0.96        45
    # Note: I'm confused that the accuracy is 100% for virginica in the matrix
        # and versicolor in this report - will ask about it on a code review

# %%

# Page 7 - Hyperparameter Tuning in Python

# Determine how many trees is best for my forest ;)

    # Create an array with commonly used quantities
numTrees = [1, 4, 5, 8, 10, 20, 50, 75, 100, 250, 500]

numTrees

# %%
# [1, 4, 5, 8, 10, 20, 50, 75, 100, 250, 500]

    # Create an empty variable to store accuracy scores for each test model
numTreesResults = []

numTreesResults

# %%
# []

    # Create a for loop to create, train, test, and confirm accuracy for a 
        # separate random forest model for each value in the array above
for num in numTrees:
    IrisForest = RandomForestClassifier(n_estimators = num, random_state = 76)
    IrisForest.fit(xTrain2, yTrain2)
    accuracy = accuracy_score(yTest2, IrisForest.predict(xTest2))
    numTreesResults.append(accuracy)
    print(num, ':', accuracy)

# %%
# 1 : 0.9111111111111111
# 4 : 0.9555555555555556
# 5 : 0.9333333333333333
# 8 : 0.9555555555555556
# 10 : 0.9777777777777777
# 20 : 0.9555555555555556
# 50 : 0.9555555555555556
# 75 : 0.9555555555555556
# 100 : 0.9555555555555556
# 250 : 0.9555555555555556
# 500 : 0.9555555555555556
    # Note: 10 trees seems like the best option!


    # Plot the accuracy scores
plt.plot(numTrees, numTreesResults)

# %%
# Note: Validates results above (though prior method is easier to interpret)


# Determine best option for remaining three hyperparameters

    # Create variables for each parameter, using common options provided by
        # lesson

        # Number of features to consider at every split
max_features = ['auto', None, 'log2']

max_features

# %%
# ['auto', None, 'log2']


        # Maximum number of levels in tree
max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, None]

max_depth

# %%
# [10, 20, 30, 40, 50, 60, 70, 80, 90, None]


        # Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

min_samples_leaf

# %%
# [1, 2, 4]

        # Method of selecting samples for training each tree
random_grid = {'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_leaf': min_samples_leaf}

random_grid

# %%
# {'max_features': ['auto', None, 'log2'],
#  'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, None],
#  'min_samples_leaf': [1, 2, 4]}


    # Create RF model to test hyperparameters with
randomForest = RandomForestClassifier(n_estimators = 10)

# %%

    # Run search with hyperparameter variables
randomForestSearch = RandomizedSearchCV(estimator = randomForest, 
    param_distributions = random_grid, n_iter = 90, cv = 3, random_state = 42)

# %%

    # Train search model
randomForestSearch.fit(xTrain2, yTrain2)

# %%
# RandomizedSearchCV(cv=3, estimator=RandomForestClassifier(n_estimators=10),
#                    n_iter=90,
#                   param_distributions={'max_depth': [10, 20, 30, 40, 50, 60,
#                                                       70, 80, 90, None],
#                                      'max_features': ['auto', None, 'log2'],
#                                         'min_samples_leaf': [1, 2, 4]},
#                    random_state=42)


    # Confirm best hyperparameter options
randomForestSearch.best_params_

# %%
# {'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 40}
    # Note: These values are different from what's in the lesson... 


    # Create and train new model with these options
newRandFor = RandomForestClassifier(n_estimators = 10, min_samples_leaf = 2, 
    max_features = 'auto', max_depth = 40)

newRandFor.fit(xTrain2, yTrain2)

# %%
# RandomForestClassifier(max_depth=40, min_samples_leaf=2, n_estimators=10)

    # Test model
newRandForPredictions = newRandFor.predict(xTest2)

# %%

# Interpret

    # Confusion matrix
print(confusion_matrix(yTest2, newRandForPredictions))

# %%
# [[19  0  0]
#  [ 0 12  1]
#  [ 0  1 12]]
    # Note: Not sure if this is better or worse than original, per now being
        # less than 100% where we had 100% before, but higher in the other 
        # that was not yet 100%...
    

    # Classification report
print(classification_report(yTest2, newRandForPredictions))

# %%
#               precision    recall  f1-score   support

#       setosa       1.00      1.00      1.00        19
#   versicolor       1.00      0.92      0.96        13
#    virginica       0.93      1.00      0.96        13

#     accuracy                           0.98        45
#    macro avg       0.98      0.97      0.97        45
# weighted avg       0.98      0.98      0.98        45
    # Note: With overall accuracy of 98%, I think this is better... it also 
        # is different from confusion matrix results again...

# %%

# Page 8 - Feature Importance

    # Find feature importance for hypertuned model
irisFeatImp = pd.Series(newRandFor.feature_importances_, index = xIris.columns)

irisFeatImp

# %%
# sepal_length    0.085567
# sepal_width     0.001547
# petal_length    0.574663
# petal_width     0.338223
# dtype: float64
    # Note: Looks like sepal width adds very little value, and petal width may
        # also be questionable

            # Return feature importance for hypertuned model in ranked order
irisFeatImp.sort_values(inplace = True, ascending = False)

irisFeatImp

# %%
# petal_length    0.574663
# petal_width     0.338223
# sepal_length    0.085567
# sepal_width     0.001547
# dtype: float64
    # Note: This was more helpful than I expected b/c I had misread numbers 
        # above!

            # Plot feature importance
irisFeatImp.plot(kind = 'barh', figsize = (7, 6))

# %%

# Page 9 - APIs

# Install Quandl API
# pip install quandl
    # Note: This didn't work for me, nor did `conda install quandl`, per syntax
        # errors, so I installed directly in Anaconda Environments, and 
        # commented this out


# Import data
alaska = quandl.get('FMAC/HPI_AK')

alaska

# %%
# 557 rows × 2 columns


# Enter API key to access data
# quandl.ApiConfig.api_key = "wytwq8oKXqFezaidUqez"
    # Note: commented this out b/c the key is from the lesson and not actually 
    # mine
