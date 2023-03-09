# %%
# DSO106 - Machine Learning and Modeling
    # Lesson 5 - Randomly Generating Data
    # Page 8 - Simulation Mastery Hands-On

# Requirements: Simulation Hands-On
# A retailer wants to create a simulation to predict the profit on the sales of 
    # a certain tool she carries. She knows the profit is a function of several 
    # factors, for which she has historical data:
        # Units Sold: Normal distribution, with a mean of 26 units and a 
            # standard deviation of 5.7 units.
        # Price: Discrete distribution. 55% of the time the price is 38     
            # dollars, 30% of the time the price is 41.50 dollars, and 15% of 
            # the time is 36.25 dollars.
        # Cost: Uniform distribution, with a max of 33.72 dollars and a min of 
            # 26.88 dollars.
        # Resource Factor: Normal distribution, with a mean of 3 and a standard 
            # deviation of 1.2.
    # The function for profit is as follows:
        # Profit = (RF * (Units sold) * (Price)) - ((0.2) * (RF) * (Units sold) 
        # * (Cost)) + $320
# Create a simulation that has 100 rows of monthly profits. Once you have 
    # completed this simulation exercise, prepare a report stating what you 
    # did, what you learned, and your results. You have the option to complete 
    # your simulation in either R, Python or Excel. Then submit it for grading.


# Import packages
import numpy as np
import pandas as pd
import random
import seaborn as sns

# %%

# Units Sold: Normal distribution, with a mean of 26 units and a standard 
    # deviation of 5.7 units.
unitsSold = (np.random.normal(size = 100)) * 5.7 + 26

unitsSold

# %%
# Note: Success!


# Price: Discrete distribution. 55% of the time the price is 38 dollars, 30% of 
    # the time the price is 41.50 dollars, and 15% of the time is 36.25 dollars.
prices = [38, 41.5, 36.25]
priceProbability = [0.55, 0.30, 0.15]
price = np.random.choice(prices, 100, p = priceProbability)

price

# %%
# Note: Success!

# Cost: Uniform distribution, with a max of 33.72 dollars and a min of 26.88 
    # dollars.
cost = []

for number in range(100):
    cost.append(random.uniform(26.88, 33.72))

cost

# %%
# Note: Success!

# Resource Factor: Normal distribution, with a mean of 3 and a standard
    # deviation of 1.2.
rf = (np.random.normal(size = 100)) * 1.2 + 3

rf

# %%
# Note: Success!

# The function for profit is as follows:
    # Profit = (RF * (Units sold) * (Price)) - ((0.2) * (RF) * (Units sold) * 
        # (Cost)) + $320
def profitFunction(rf, unitsSold, price, cost):
    return (rf * unitsSold * price) - ((0.2) * rf * unitsSold * cost) + 320

roundFunction = np.vectorize(lambda x: "%.2f" % x)

profit = profitFunction(rf, unitsSold, price, cost)

print(roundFunction(profit))

# %%
# Note: Success!

# Convert profit to a dataframe to export
profitExport = pd.DataFrame({'Profit': roundFunction(profit)})

profitExport

# %%
# 100 rows × 1 columns
    # Note: Success!

    # Export to csv
profitExport.to_csv('/Users/hannah/Library/CloudStorage/GoogleDrive-gracesnouveaux@gmail.com/My Drive/Bethel Tech/Data Science/DSO106 Machine Learning and Modeling/1: Modeling and Optimization – Lesson 5. Randomly Generating Data/profitsPython.csv')

# %%

# Histogram
sns.histplot(profit)

# %%
# Decently normally distributed!

# Find first and third quartiles
profitQuartiles = np.percentile(profit, [25, 75])

# Print the results
print('1st Quartile:', profitQuartiles[0])
print('3rd Quartile:', profitQuartiles[1])

# %%
# 1st Quartile: 1963.6189679502922
# 3rd Quartile: 3437.919857624861
    # Note: Profits will most likely be between $1,964 and $3,438
        # These values will change each time the script is run, per the 
        # randomization of the values generated