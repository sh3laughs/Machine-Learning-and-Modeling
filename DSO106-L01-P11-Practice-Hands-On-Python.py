# %%
# DSO106 - Machine Learning and Modeling
    # Lesson 1  - Modeling with Linear Regression
    # Page 11 - Best Fit Line Hands-On

# Requirements: It is a well known phenomena that most of us shrink throughout 
    # the day each day. The effects of gravity cause that our height measured 
    # at the end of the day is less than our height measured at the beginning 
    # of the day. Fortunately, at night, our bodies stretch out again, so that 
    # from one morning to the next, each of us has returned to the morning 
    # height from the day before.
  # In the dataset below, there are AM and PM height measurements (in mm) for 
    # students from a boarding school in India.
  # Hands on Part 2: Take the above dataset, and complete simple linear 
    # regression in Python. Make sure to test, note, and correct for all 
    # assumptions if possible!

# Goal: Predict height
  # IV (x axis, continuous): AM_Height
  # DV (y axis, continuous): PM_Height
  # H0: Morning height does not predict evening height (aka: b1 = 0)
    # H1: Morning height does predict evening height (aka: b1 ≠ 0)

# Import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.api as sms
from scipy.stats import boxcox

# %%

# Import & preview data
height = pd.read_csv('/Users/hannah/Library/CloudStorage/GoogleDrive-gracesnouveaux@gmail.com/My Drive/Bethel Tech/Data Science/DSO106 Machine Learning and Modeling/1: Modeling and Optimization – Lesson 1. Modeling with Linear Regression/heights.csv')

height

# %%
# 40 rows x 2 columns

# Create linear regression model
heightLm = sm.OLS(height.PM_Height, height.AM_Height).fit()

heightLm

# %%
# <statsmodels.regression.linear_model.RegressionResultsWrapper at 
    # 0x7f90d823fa30>


# Test Assumptions

    # Linearity AND Normality
sns.pairplot(height)

# %%
# Note: These are very linear... and appear to be normally distributed

        # Normality, take 2
sns.distplot(height.AM_Height)

# %%
# Note: This is roughly normal distribution

sns.distplot(height.PM_Height)

# %%
# Note: This is also roughly normal distribution


    # Homoscedasticity

        # Calculate residuals
heightPred = heightLm.fittedvalues.copy()
heightTrue = height.PM_Height.values.copy()
heightResid = heightTrue - heightPred

        # Graph the residuals
fig, ax = plt.subplots()
_ = ax.scatter(heightResid, heightPred)

# %%
# Note: These data appear to be heteroscedastic... though in R that wasn't the
    # case...

        # Run a Breusch Pagan test
sms.diagnostic.het_breuschpagan(heightResid, height[['AM_Height']])

# %%
# (17.281807808595943, nan, 29.145236144699442, 3.3000372410769865e-06)
    # Note: The p value is significant and thus violates this assumption, 
        # aligned with the scatterplot results above (but contradicting R...)

        # Run a Harvey Collier test
sms.linear_harvey_collier(heightLm)

# %%
# Ttest_1sampResult(statistic=-0.004476102027705233, pvalue=0.9964526463756713)
    # Note: This p value is significant, which validates the assumption...


        # Correct for heteroscedasticity and re-plot to test again
heightTransformed, _ = boxcox(height.AM_Height)

plt.hist(heightTransformed)

# %%
# Note: Success! Roughly normal distribution

        # Create a new model with the transformed data
heightLm2 = sm.OLS(height.AM_Height, heightTransformed).fit()

# %%

    # Outliers

        # Influential - via plot
fig, ax = plt.subplots()

fig = sm.graphics.influence_plot(heightLm, alpha = 0.05, ax = ax)

# %%
# Note: Rows 4 and 36 are outliers


        # Influential - via table
heightInfluen = heightLm.get_influence()
print(heightInfluen.summary_frame())

# %%
# Note: Possibly row 40... or rows that are hidden from view ;)


# %%

# Run analysis on original model
heightLm.summary()

# %%
# Note: The overall p value is significant; reject the null and accept the 
    # alternative hypothesis - morning height does predict night height (that's
    # a rhyme!)
    # According to the adjusted R value, morning height explains ~100% of the
    # variance in evening height!


# Run analysis on transformed model
heightLm2.summary()

# %%
# Note: Little difference from original data, though in this model morning 
    # height only explains ~98% of variance in evening height ;)