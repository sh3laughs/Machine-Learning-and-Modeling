# DSO106 - Machine Learning and Modeling
  # Lesson 5 - Randomly Generating Data
  # Page 8 - Simulation Mastery Hands-On

# Setup ----
# A retailer wants to create a simulation to predict the profit on the sales of 
  # a certain tool she carries. She knows the profit is a function of several 
  # factors, for which she has historical data:
    # Units Sold: Normal distribution, with a mean of 26 units and a standard 
      # deviation of 5.7 units.
    # Price: Discrete distribution. 55% of the time the price is 38 dollars, 
      # 30% of the time the price is 41.50 dollars, and 15% of the time is 
      # 36.25 dollars.
    # Cost: Uniform distribution, with a max of 33.72 dollars and a min of 
      # 26.88 dollars.
    # Resource Factor: Normal distribution, with a mean of 3 and a standard 
      # deviation of 1.2.
  # The function for profit is as follows:
    # Profit = (RF * (Units sold) * (Price)) - ((0.2) * (RF) * (Units sold) * 
      # (Cost)) + $320
# Create a simulation that has 100 rows of monthly profits. Once you have 
# completed this simulation exercise, prepare a report stating what you did, 
# what you learned, and your results. You have the option to complete your 
# simulation in either R, Python or Excel. Then submit it for grading.


# Import package
library(stats)


# Units Sold ----
  # Normal distribution, with a mean of 26 units and a standard deviation of 
  # 5.7 units.
unitsSold = rnorm(100, 26, 5.7)
# Note: New variable in environment with 100 values


# Price ----
  # Discrete distribution. 55% of the time the price is 38 dollars, 30% of the 
  # time the price is 41.50 dollars, and 15% of the time is 36.25 dollars.
prices = c(38, 41.5, 36.25)
priceProbabilities = c(0.55, 0.30, 0.15)
price = sample(prices, size = 100, replace = TRUE, prob = priceProbabilities)
# Note: New variable in environment with 100 values


# Cost ----
  # Uniform distribution, with a max of 33.72 dollars and a min of 26.88 
  # dollars.
cost = runif(100, 26.88, 33.72)
# Note: New variable in environment with 100 values


# Resource Factor ----
  # Normal distribution, with a mean of 3 and a standard deviation of 1.2.
rf = rnorm(100, 3, 1.2)
# Note: New variable in environment with 100 values

# The function for profit ---
  # Profit = (RF * (Units sold) * (Price)) - ((0.2) * (RF) * (Units sold) * 
  # (Cost)) + $320
profit = (rf * unitsSold * price) - (0.2 * rf * unitsSold * cost) + 320

profit = round(profit, 2)

print(profit)
# Note: Success!


# Convert profit to dataframe to export ----

  # Convert
profits = as.data.frame(profit)

View(profits)
# 100 entries, 1 total columns


  # Export
write.csv(profits, '/Users/hannah/Library/CloudStorage/GoogleDrive-gracesnouveaux@gmail.com/My Drive/Bethel Tech/Data Science/DSO106 Machine Learning and Modeling/1: Modeling and Optimization – Lesson 5. Randomly Generating Data/profitsR.csv')
# Note: Success!


# Create histogram ----
hist(profit)
# Note: Decently normally distributed!


# Find quartiles ----
summary(round(profit))
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# -244    1710    2613    2738    3305    7104 
  # Note: Profits will most likely be between $1,710 and $3,305
    # These values will change each time the script is run, per the 
    # randomization of the values generated, unless the csv exported is used
    # as a starting point