# %%
# DSO106 - Machine Learning and Modeling
    # Lesson 2  - Modeling with Logistic Regression

# Page 6 - Logistic Regression in Python

# Goal: Determine whether home runs predict a winning game score
  # IV (x axis, continuous): home runs
  # DV (y axis, categorical): winning game score
  # H0: Home runs do not predict a winning game score
    # H1: Home runs do predict a winning game score

# Import packages
import pandas as pd
import statsmodels.api as sm

# %%

# Import and preview data
baseball = pd.read_csv('/Users/hannah/Library/CloudStorage/GoogleDrive-gracesnouveaux@gmail.com/My Drive/Bethel Tech/Data Science/DSO106 Machine Learning and Modeling/1: Modeling and Optimization – Lesson 2. Modeling with Logistic Regression/baseball.csv')

baseball

# %%
# 4860 rows × 10 columns

# Wrangling

    # Recode DV
def recodeDV (series):
    if series == 'W':
        return 1
    if series == 'L':
        return 0

baseball['winsR'] = baseball['W/L'].apply(recodeDV)

baseball

# %%
# 4860 rows × 11 columns

# Run analysis

    # Create logit
baseballLogit = sm.Logit(baseball.winsR, baseball['HR Count'])

# %%
    
    # Run analysis
baseballLogitResults = baseballLogit.fit()

print(baseballLogitResults.summary2())

# %%
# Note: The p value for HR's is significant, which means they do predict wins;
    # reject the null and accept the alternative hypothesis
    # For every home run, the odds of winning increases by ~28%... this seems
    # more realistic than the ~66% results in R, but I'm curious about that big
    # difference!
    # Home runs account for ~4% of the variance in predicting a winning score
   
# %%

# Testing to see if I get different results if I store the IV and DV in their
    # own variables, as lesson dictates
x = baseball['HR Count']
y = baseball.winsR

baseballLogit2 = sm.Logit(y, x)

baseballLogitResults2 = baseballLogit2.fit()

print(baseballLogitResults2.summary2())

# %%
# Note: Same results
    # I had tested this in the prior lesson in this course, but quiz 
    # specifically said on this page these variables are required.. which is
    # still not true...