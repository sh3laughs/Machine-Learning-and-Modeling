# %%
# DSO106 - Machine Learning and Modeling
    # Lesson 1  - Modeling with Linear Regression

# Page 8 - Regression Setup in Python

# Goal: Determine whether the number of powerboats registered in Florida 
    # impacts the number of manatees killed there by powerboats each year
  # IV (x axis, continuous): powerboats
  # DV (y axis, continuous): manatee deaths
  # H0: The number of powerboats registered in FL does not predict the number
      # of manatees killed there by powerboats each year (aka: b1 = 0)
    # H1: The number of powerboats registered in FL does predict the number of
      # manatees killed there by powerboats each year (aka: b1 ≠ 0)

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
manatees = pd.read_csv('/Users/hannah/Library/CloudStorage/GoogleDrive-gracesnouveaux@gmail.com/My Drive/Bethel Tech/Data Science/DSO106 Machine Learning and Modeling/1: Modeling and Optimization – Lesson 1. Modeling with Linear Regression/manatees.csv')

manatees

# %%
# 35 rows x 2 columns

# Test Assumptions

    # Linearity AND Normality
sns.pairplot(manatees)

# %%
# Note: These are roughly linear... don't appear to be normally distributed, 
    # but will try another test for that

        # Normality, take 2
sns.distplot(manatees.PowerBoats)

# %%
# Note: This is roughly normal distribution

sns.distplot(manatees.ManateeDeaths)

# %%
# Note: This is also roughly normal distribution


    # Homoscedasticity

        # Create linear regression model
manateesLm = sm.OLS(manatees.ManateeDeaths, manatees.PowerBoats).fit()

manateesLm

# %%
# <statsmodels.regression.linear_model.RegressionResultsWrapper at 
    # 0x7fe59a9a6ca0>

        # Calculate residuals
manateesPred = manateesLm.fittedvalues.copy()
manateesTrue = manatees.ManateeDeaths.values.copy()
manateesResid = manateesTrue - manateesPred

        # Graph the residuals
fig, ax = plt.subplots(figsize = (6, 2.5))
_ = ax.scatter(manateesResid, manateesPred)

# %%
# Note: These data are heteroscedastic - and violate this assumption

            # Testing code with no assigned size
fig, ax = plt.subplots()
_ = ax.scatter(manateesResid, manateesPred)
# %%
# Note: This works just as well...

        # Run a Breusch Pagan test
sms.diagnostic.het_breuschpagan(manateesResid, manatees[['PowerBoats']])

# %%
# (14.786518579352656, nan, 24.8716003560108, 1.7849503580150725e-05)
    # Note: The last # is the p value, which is significant and thus violates
        # this assumption, aligned with the scatterplot results above
    # Also noteworthy... this code only works with double brackets around the 
        # IV column name...

        # Run a Harvey Collier test
sms.linear_harvey_collier(manateesLm)

# %%
# Ttest_1sampResult(statistic=5.479252885631467, pvalue=5.433881942347164e-06)
    # Note: This p value is also significant, which also violates the 
        # assumption and alignes with results above


        # Correct for heteroscedasticity and re-plot to test again
manateesTransformed, _ = boxcox(manatees.PowerBoats)

plt.hist(manateesTransformed)

# %%
# Note: This didn't work - data is still not normally distributed, and thus
    # violates this assumption still... but we're going forward anyway
    # ALSO, I thought it was weird that we were boxcoxing our IV here, but our
    # DV in R, so I asked about that in a code review and confirmed this should
    # have been the DV


        # Just for the sake of practice, create a new model with the transformed
            # data
manateesLm2 = sm.OLS(manatees.ManateeDeaths, manateesTransformed).fit()

manateesLm2.summary()

# %%
# Note: Output we're not yet taught how to interpret...

        # Continued practice by plotting transformed residuals
manateesPred2 = manateesLm2.fittedvalues.copy()
manateesTrue2 = manatees.ManateeDeaths.values.copy()
manateesResid2 = manateesTrue2 - manateesPred2

fig, ax = plt.subplots()
_ = ax.scatter(manateesResid2, manateesPred2)

# %%
# Note: As we knew it would, it still violates the assumption, and yet the show
    # goes on...


        # And for more practice.. run a correlation, though it's not actually
            # needed unless there are more IV's
manatees.corr()

# %%

        # Plot that correlation
sns.heatmap(manatees.corr(), annot =  True)

# %%


    # Outliers

        # Influential - via plot
fig, ax = plt.subplots()

fig = sm.graphics.influence_plot(manateesLm, alpha = 0.05, ax = ax)

# %%
# Note: No row numbers on dots means no influential outlier identified


        # Influential - via table
manateesInfluen = manateesLm.get_influence()
print(manateesInfluen.summary_frame())

# %%
# Note: Whew! Big table!

    # Note: Apparently we're not learning how to test for distance or leverage
        # outliers in Python...


# %%

# Run analysis
manateesLm.summary()

# %%
# Note: The p value (Prob - F-statistic) is significant
    # That said, other violations are in these data, so it's not a reliable
    # test