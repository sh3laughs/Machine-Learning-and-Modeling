# DSO106 - Machine Learning and Modeling
  # Lesson 2 - Modeling with Logistic Regression
  # Page 8 - Modeling with Logistic Regression Hands-On

# Setup ----

# Requirements: Geologists have long known that the presence of antimony in a 
    # ground sample can be a good indication that there is a gold deposit 
    # nearby. In the attached data, you will find antimony (Sb) levels for 64 
    # different locations, and whether or not gold was found nearby. The "gold" 
    # column is coded as 0 for no gold present, and 1 for gold present.
  # Use logistic regression in R to create a prediction model that will give 
  # the probability of the presence of gold as a response. Your complete report 
  # should contain the following information:
    # 1. Testing and correction for assumptions if necessary.
    # 2. Interpretation of the results in layman's terms.

# Goal: Determine whether the presence of antimony predicts the presence of gold
  # IV (x axis, continuous): antimony
  # DV (y axis, categorical): gold
  # H0: The presence of antimony does not predict the presence of gold
    # H1: The presence of antimony does predict the presence of gold


# Import potential packages
library(caret)
library(dplyr)
library(e1071)
library(ggplot2)
library(IDPmisc)
library(lmtest)
library(magrittr)
library(popbio)
library(tidyr)

# Import and preview data
gold = read.csv('/Users/hannah/Library/CloudStorage/GoogleDrive-gracesnouveaux@gmail.com/My Drive/Bethel Tech/Data Science/DSO106 Machine Learning and Modeling/1: Modeling and Optimization – Lesson 2. Modeling with Logistic Regression/minerals.csv')

View(gold)
# 64 entries, 2 total columns
  # Note: The DV is already coded as 0's and 1's


# Prep ----

# Create logistic regression model and prep data for testing assumptions

  # Create model
goldLogM = glm(Gold ~ Antimony, data = gold, family = 'binomial')


  # Create probabilities variable then add to dataframe as converted to 
    # positive / negative predictions (aka: responses or probable outcomes)
goldProb = predict(goldLogM, type = 'response')
gold$predicted = if_else(goldProb > 0.5, 'pos', 'neg')

View(gold)
# 64 entries, 3 total columns


  # Recode predictions variable
gold$predictedR = NA
gold$predictedR[gold$predicted == 'pos'] = 1
gold$predictedR[gold$predicted == 'neg'] = 0

View(gold)
# 64 entries, 4 total columns


  # Convert predictions and DV to factors
gold$predictedR = as.factor(gold$predictedR)
gold$Gold = as.factor(gold$Gold)

str(gold)
# Note: Success! Both are factors, with 2 levels each



# Test assumptions ----

  # Sample size
goldCm = caret::confusionMatrix(gold$predictedR, gold$Gold)

goldCm
# Confusion Matrix and Statistics
#           Reference
# Prediction  0  1
#          0 34  8
#          1  2 20
#                Accuracy : 0.8438
  # Note: Sample size validated! All cells have at least one case, and only 1
    # (<20%) has less than 5


  # Logit linearity

    # Create new dataframe with only IV
gold2 = as.data.frame(gold$Antimony)

View(gold2)

    # Create new dataframe to include a new column with the logit (calculated 
      # in this line of code using dataframe just created above), as well as 
      # two columns for the predictor (IV) name and its values
gold3 = gold2 %>% 
  mutate(goldLogit = log(goldProb / (1 - goldProb))) %>% 
  gather(key = 'goldPredictors', value = 'goldPredictor.value', 
         -goldLogit)

View(gold3)
# 256 entries, 3 total columns

    # Graph the logit
ggplot(gold3, aes(goldLogit, goldPredictor.value)) +
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = 'loess') +
  theme_bw() +
  facet_wrap(~goldPredictors, scales = 'free_y')
# Note: Validated! Very linear


  # Multicollinearity: N/A per only one IV


  # Independent errors
plot(goldLogM$residuals)
# Note: A nice distribution along the x axis - validated!


  # Outliers - influential
goldInfluencers = influence.measures(goldLogM)

summary(goldInfluencers)
# Note: Validated! No dfb.1_ or dffit values over 1 and no hat values over 0.3



# Run analysis ----
summary(goldLogM)
# ...
# Coefficients:
#             Estimate Std. Error z value Pr(>|z|)    
# (Intercept)  -2.5184     0.5958  -4.227 2.37e-05 ***
# Antimony      1.7606     0.4883   3.606 0.000311 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# ...
  # Note: This p value indicates that the presence of antimony significantly 
    # predicts the presence of gold; reject the null and accept the alternative 
    # hypothesis - for each 1 unit increase in antimony, there is an expected 
    # 1.7606 increase in gold


# Summary ----
# The presence of antimony is a significant predictor of the presence of gold;
  # for each 1 unit increase in antimony, there is an expected 1.7606 increase 
  # in gold