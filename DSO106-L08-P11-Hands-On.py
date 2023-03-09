# %%
# DSO106 - Machine Learning and Modeling
    # Lesson 8 - Decision Trees and Random Forests
        # AKA: Machine Learning Lesson 3
    # Page 11 - Lesson 3 Hands-On

# Requirements: Part I
    # Create a decision tree model of the Titanic dataset that predicts 
        # survival from seaborn. You will need to import the data using this 
        # code:
            # Titanic = sns.load_dataset('titanic')
    # You will need to compute some data wrangling before charging ahead. Make 
        # sure to complete the following wrangling tasks:
            # Recode string data
            # Remove missing data
            # Drop any variables that are redundant and will add to 
                # multicollinearity.
    # Once you have created a decision tree model, interpret the confusion 
        # matrix and classification report.

# Goal: Determine whether any / all variables predict survival
    # IV's (x axis, continuous): TBA
    # DV (y axis, categorical): survival
    # H0: No variables predict survival
        # H1: At least one variable predicts survival

# Import potential packages (for both parts I and II)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# %%

# Import data and exploration 
titanicOriginal = sns.load_dataset('titanic')

titanicOriginal

# %%
# 891 rows × 15 columns
    # Note: `survived` is DV
        # Lots of string data! Recode these (if kept):
            # sex, embarked, class, who, adult_male, deck, embark_town, alive
                # alone
        # Variables that are / are possibly redundant
            # `pclass` appears to be an already recoded version of the 
                # `class`, so I'll keep that and drop class, which is string
            # `who` seems silly, since we have `sex`, so I'll probably drop it
                # but will check its values first
            # `adult_male` seems redundant with `sex` and `age`, or possibly 
                # `who`, but only if the ages only include adults or `who` 
                # includes children
            # `embark_town` seems to be a longer string version of `embarked`
                # will confirm, and drop one, if so
            # `alive` appears to be a string version of `survived`, so I'll 
                # drop it


    # Confirm whether `who` has any values other than `man` and `woman`
titanicOriginal.who.value_counts()

# %%
# man      537
# woman    271
# child     83
# Name: who, dtype: int64
    # Note: Hmm.. so we definitely don't need this and adult_male! Though 
        # having age, also, may be useful – I'll drop adult_male and keep who


    # Confirm values for `embarked`
titanicOriginal.embarked.value_counts()

# %%
# S    644
# C    168
# Q     77
# Name: embarked, dtype: int64


        # Confirm values for `embark_town`
titanicOriginal.embark_town.value_counts()

# %%    
# Southampton    644
# Cherbourg      168
# Queenstown      77
# Name: embark_town, dtype: int64
    # Note: These two are the same! So I'll drop one and recode the other


    # Find sums of missing values per column
titanicOriginal.isna().sum()

# %%
# survived         0
# pclass           0
# sex              0
# age            177
# sibsp            0
# parch            0
# fare             0
# embarked         2
# class            0
# who              0
# adult_male       0
# deck           688
# embark_town      2
# alive            0
# alone            0
# dtype: int64
    # Note: `deck` has so many missing values that it would be a sad loss to
        # the model to remove NA's and lose 688 rows, so I'll just delete that
        # column


# Wrangling

    # Drop redundant variables + `deck`, per it's missing values
titanic1 = titanicOriginal
titanic1.drop(['class', 'adult_male', 'deck', 'embark_town', 'alive'], 
    axis = 1, inplace = True)

titanic1

# %%
# 891 rows × 10 columns
    # Note: Success!


    # Recode string IV's: sex, embarked, who, embark_town, alone

        # sex - using replace function
titanic1['sexR'] = titanic1.sex

titanic1.sexR.replace(['male', 'female'], [0, 1], inplace = True)

titanic1.drop(['sex'], axis = 1, inplace = True)

titanic1

# %%
# 891 rows × 10 columns
#   Note: Success! Though I'm not sure if the data type will be correct with 
        # this function, so I picked this method to be able to go through all 
        # steps and see if I prefer it over the `def`ine a function method

            # Confirm data type for new sexR variable
titanic1.info()

# %%
# Note: Success! It's int64.. so yeah, I like this method ;)


        # embarked - recode using def function function 8) (<that is the nerd 
            # emoji, which is my favorite)
def embarkedRecode (value):
    if value == 'S':
        return 0
    if value == 'C':
        return 1
    if value == 'Q':
        return 2

titanic1['embarkedR'] = titanic1.embarked.apply(embarkedRecode)

titanic1.drop(['embarked'], axis = 1, inplace = True)

titanic1

# %%
# 891 rows × 10 columns
    # Note: The one funny thing is that it recoded as floats (ie: 1.0), and 
        # I'm not sure why.. though at least that validates it's numeric


        # who - using replace function again!
titanic1['whoR'] = titanic1.who

titanic1.whoR.replace(['man', 'woman', 'child'], [0, 1, 2], inplace = True)

titanic1.drop(['who'], axis = 1, inplace = True)

titanic1

# %%
# 891 rows × 10 columns


        # alone - recode using replace
titanic1['aloneR'] = titanic1.alone

titanic1.aloneR.replace([False, True], [0, 1], inplace = True)

titanic1.drop(['alone'], axis = 1, inplace = True)

titanic1

# %%
# 891 rows × 10 columns
    # Note: Voila! We are fully recoded


    # Drop missing values
titanic = titanic1.copy()

titanic.dropna(inplace = True)

titanic

# %%
# 712 rows × 10 columns


    # Define x variable for IV's
xColumns = list(range(1, 10))
x = titanic.iloc[:, xColumns]

x

# %%
# 712 rows × 9 columns


    # Define y variable for DV
y = titanic.survived

y

# %%
# 0      0
# 1      1
# 2      1
# 3      1
# 4      0
#       ..
# 885    0
# 886    0
# 887    1
# 889    1
# 890    0
# Name: survived, Length: 712, dtype: int64


    # Train / test split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3)

# %%
# Note: We're wrangled!


# Create decision tree model
titanicDtree = DecisionTreeClassifier()

# %%

    # Train the model
titanicDtree.fit(xTrain, yTrain)

# %%
# DecisionTreeClassifier()

    # Test the model
titanicDtreePredictions = titanicDtree.predict(xTest)

# %%


# Interpret

    # View confusion matrix (labeled, for legibility)
survivedDtreeValues = [0, 1]

titanicDtreeCmInitial = confusion_matrix(yTest, titanicDtreePredictions, 
    labels = survivedDtreeValues)

titanicDtreeCm = pd.DataFrame(titanicDtreeCmInitial, 
    index = survivedDtreeValues, columns = survivedDtreeValues)

titanicDtreeCm

# %%
#   0	1
# 0	98	22
# 1	33	61
    # Note: The model is more accurate than not ;)
        # It is most accurate at successfully predicting fatalities (98) :(
        # It is also accurate at predicting survivals (61)
        # I'm also now realizing these values don't sum to 712, for the total
            # # of rows in the dataset, which I would have expected... 
    

    # View classification report
print(classification_report(yTest, titanicDtreePredictions))

# %%
#               precision    recall  f1-score   support
#            0       0.75      0.82      0.78       120
#            1       0.73      0.65      0.69        94
#     accuracy                           0.74       214
#    macro avg       0.74      0.73      0.74       214
# weighted avg       0.74      0.74      0.74       214
    # Note: This confirms that the model is better at predicting fatalities
        # (75%) than it is at predicting survival (73%), and is pretty decent,
        # at 74% overall accuracy!

# Decision Tree Summary
    # If we stopped here, we could safely say no one would want to rely on 
    # this model to tell them what their best chances for survival are during 
    # a ship wreck...

# %%

# Requirements: Part II
    # Now create a random forest model of the Titanic dataset that predicts 
        # survival. Interpret the confusion matrix and classification report. 
        # How did the predictive value change from the decision tree?


# Create initial random forest model
    # Note: Using random_state for reproducability while grading ;)
titanicRF = RandomForestClassifier(random_state = 101)

# %%

    # Train model
titanicRF.fit(xTrain, yTrain)

# %%
# RandomForestClassifier(random_state=101)


    # Run model
titanicRFPredictions = titanicRF.predict(xTest)

# %%

# Interpret

    # View confusion matrix (labeled, for legibility)
survivedRFValues = [0, 1]

titanicRFCmInitial = confusion_matrix(yTest, titanicRFPredictions, 
    labels = survivedRFValues)

titanicRFCm = pd.DataFrame(titanicRFCmInitial, index = survivedRFValues, 
    columns = survivedRFValues)

titanicRFCm

# %%
#   0	1
# 0	97	23
# 1	28	66
    # Note: This is a miniscule decrease in correctly predicting fatalaties, and
        # a small increase in correctly predicting survivals - so overall, it's
        # a more accurate model!


    # Classification report
print(classification_report(yTest, titanicRFPredictions))

# %%
#               precision    recall  f1-score   support
#            0       0.78      0.81      0.79       120
#            1       0.74      0.70      0.72        94
#     accuracy                           0.76       214
#    macro avg       0.76      0.76      0.76       214
# weighted avg       0.76      0.76      0.76       214
    # Note: Contradicting the confusion matrix, the fatality predictions have a
        # higher increase in accuracy than survivals, though both are improved
        # ... if things stopped here, the model has an overall accuracy of 76%,
        # which is pretty good


# Hyperparameter Tuning

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
    titanicRFnEst = RandomForestClassifier(n_estimators = num, 
        random_state = 101)
    titanicRFnEst.fit(xTrain, yTrain)
    accuracy = accuracy_score(yTest, titanicRFnEst.predict(xTest))
    numTreesResults.append(accuracy)
    print(num, ':', accuracy)

# %%
# 1 : 0.7757009345794392
# 4 : 0.7757009345794392
# 5 : 0.7850467289719626
# 8 : 0.7897196261682243 - winner, winner, chicken dinner
# 10 : 0.7757009345794392
# 20 : 0.7616822429906542
# 50 : 0.7570093457943925
# 75 : 0.7616822429906542
# 100 : 0.7616822429906542
# 250 : 0.7663551401869159
# 500 : 0.7663551401869159


    # Determine best option for remaining three hyperparameters

        # Create variables for each parameter, using common options
max_features = ['auto', None, 'log2']
max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, None]
min_samples_leaf = [1, 2, 4]

print(max_features)
print(max_depth)
print(min_samples_leaf)

# %%
# ['auto', None, 'log2']
# [10, 20, 30, 40, 50, 60, 70, 80, 90, None]
# [1, 2, 4]


        # Create grid of options
random_grid = {'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_leaf': min_samples_leaf}

random_grid

# %%
# {'max_features': ['auto', None, 'log2'],
#  'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, None],
#  'min_samples_leaf': [1, 2, 4]}


        # Create RF model to test hyperparameters with
titanicRFtuning = RandomForestClassifier(n_estimators = 8, random_state = 101)

# %%

        # Run search with hyperparameter variables
titanicRFSearch = RandomizedSearchCV(estimator = titanicRFtuning, 
    param_distributions = random_grid, n_iter = 90, cv = 5, 
    random_state = 101)

# %%

        # Train search model
titanicRFSearch.fit(xTrain, yTrain)

# %%
# RandomizedSearchCV(cv=5,
#                    estimator=RandomForestClassifier(n_estimators=8,
#                                                     random_state=101),
#                    n_iter=90,
#                    param_distributions={'max_depth': [10, 20, 30, 40, 50, 60,
#                                                       70, 80, 90, None],
#                                        'max_features': ['auto', None, 'log2'],
#                                         'min_samples_leaf': [1, 2, 4]},
#                    random_state=101)


         # Confirm best hyperparameter options
titanicRFSearch.best_params_

# %%
# {'min_samples_leaf': 4, 'max_features': None, 'max_depth': 10}


    # Create and train new model with these options
titanicRFtuned = RandomForestClassifier(n_estimators = 8, min_samples_leaf = 4, 
    max_features = None, max_depth = 10)

titanicRFtuned.fit(xTrain, yTrain)

# %%
# RandomForestClassifier(max_depth=10, max_features=None, min_samples_leaf=4,
#                        n_estimators=8)

    # Test model
titanicRFtunedPredictions = titanicRFtuned.predict(xTest)

# %%

# Interpret

    # Confusion matrix
survivedRFTunedValues = [0, 1]

titanicRFTunedCmInitial = confusion_matrix(yTest, titanicRFtunedPredictions, 
    labels = survivedRFTunedValues)

titanicRFTunedCm = pd.DataFrame(titanicRFTunedCmInitial, 
    index = survivedRFTunedValues, columns = survivedRFTunedValues)

titanicRFTunedCm

# %%
#     0	1
# 0	101	19
# 1	30	64
    # Note: The number of accurately predicted fatalities went up noticeably,   
        # though the number of accurately predicted survivals went down... :(
    

    # Classification report
print(classification_report(yTest, titanicRFtunedPredictions))

# %%
#               precision    recall  f1-score   support
#            0       0.77      0.84      0.80       120
#            1       0.77      0.68      0.72        94
#     accuracy                           0.77       214
#    macro avg       0.77      0.76      0.76       214
# weighted avg       0.77      0.77      0.77       214
    # Note: Well, here it says that accuracy for fatalities decreased and 
        # accuracy for survivals increased, so that they are now both 77%... so 
        # I really need to learn why the confusion (indeed!) matrices and 
        # classification reports seem to tell two different stories.. overall 
        # accuracy is also 77% hmm...
    

    # Find feature importance for hypertuned model
titanicFeatImp = pd.Series(titanicRFtuned.feature_importances_, 
    index = x.columns)

titanicFeatImp.plot(kind = 'barh', figsize = (10, 5))

# %%
# Note: Sex is the highest predictor, which is not surprising given known   
        # history
    # Age, who (which is like a combination of age and sex), fare, and
        # passenger class are all similar as next-level predictors – though 
        # with about half of the impact on predictions as sex
    # Sibsp (not sure what this is!) has enough of an effect that it seems
        # good that it is included
    # Alone, embarked, and parch (also not sure what this one is!) have a low
        # enough effect that the model may be fine (or improved?) without them


# Random Forest Summary
    # Using a hypertuned random forest model improved the accuracy by 3% 
    # overall, as well as by 2% and 4% for predicting fatalites and survivals,
    # respectively. This model seems useful, though I would not want to trust it
    # with my life!