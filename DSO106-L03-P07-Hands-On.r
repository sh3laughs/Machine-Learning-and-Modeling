# DSO106 - Machine Learning and Modeling
  # Lesson 3 - Non-Linear Modeling
  # Page 7 - Lesson 3 Hands-On

# Setup ----

# Requirements: Data from the following spreadsheet will be used throughout this 
    # hands on. You have two sets of X and Y variables here; graph and analyze 
    # both and determine what non-linear form they best follow. These two sets 
    # of X and Ys might both be exponential relationships or quadratic 
    # relationships, or there might be one of each. The best way to figure it 
    # out is to try and fit both a quadratic function and an exponential 
    # function to each pair of variables, and then model each to determine which 
    # model is a better fit.
  # To complete this hands on, you will need to:
    # 1. Create a scatterplot of the data with the Y variable on the vertical 
      # axis, and the X variable on the horizontal axis.
    # 2.Using eyeball analysis, make a guess about what type of model will work 
      # best for the dataset. You can add the best fit quadratic line as well 
      # to determine if it's a good fit.
    # 3. Using the chosen model from step 2, complete the steps to perform the 
      # analysis that were listed in the lesson.

# Goal1: Determine whether X1 predicts Y1
  # IV (x axis, continuous): X1
  # DV (y axis, categorical): Y1
  # H0: X1 does not predict Y1
    # H1: X1 does predict Y1

# Goal2: Determine whether X2 predicts Y2
  # IV (x axis, continuous): X2
  # DV (y axis, categorical): Y2
  # H0: X2 does not predict Y2
    # H1: X2 does predict Y2


# Import package
library(ggplot2)


# Import and preview data
unknown = read.csv('/Users/hannah/Library/CloudStorage/GoogleDrive-gracesnouveaux@gmail.com/My Drive/Bethel Tech/Data Science/DSO106 Machine Learning and Modeling/1: Modeling and Optimization – Lesson 3. Non-Linear Modeling/nonlinear.csv')

View(unknown)
# 39 entries, 4 total columns


# Graph data to confirm relationships ----

  # Graph data '1' as exponential
ggplot(unknown, aes(x = X1, y = Y1)) +
  geom_point() + 
  stat_smooth(method = 'lm', formula = y ~ x + I(exp(x)), size = 1)
# Note: This is tricky... I think this is quadratic b/c the data doesn't quite
  # align with the best fit exponential line - though the data does have a 
  # noticeable decrease as it moves across the x axis, without a noticeable 
  # prior decrease, it does have a very sligh increase before the decrease 
  # begins... hard to call this parabolic, though...


  # Graph data '1' as quadratic
ggplot(unknown, aes(x = X1, y = Y1)) +
  geom_point() + 
  stat_smooth(method = 'lm', formula = y ~ x + I(x ^ 2), size = 1)
# Note: The data do align with the quadratic best fit line more than the 
  # exponential, so I am going to consider these variables as having a 
  # quadratic relationship...


  # Graph data '2' as exponential
ggplot(unknown, aes(x = X2, y = Y2)) +
  geom_point() + 
  stat_smooth(method = 'lm', formula = y ~ x + I(exp(x)), size = 1)
# Note: These data do appear to have an exponential relationship - the data
  # has a noticeable increase as it moves across the x axis, without a prior
  # increase (to make it parabolic), and they align with the best fit line

  # Graph data '2' as quadratic
ggplot(unknown, aes(x = X2, y = Y2)) +
  geom_point() + 
  stat_smooth(method = 'lm', formula = y ~ x + I(x ^ 2), size = 1)
# Note: The best fit line difference is subtle, but these data do align better
  # with the exponential best fit line, so I am going to consider these 
  # variables as exponentially related


# Run analysis on '1' data ----
  # Note: Because these data were not distinctly quadratic, I will run both
    # types of regression

  # Non-linear regression for an exponential relationship
unknownExM1 = lm(log(unknown$Y1) ~ unknown$X1)

summary(unknownExM1)
# Call:
# lm(formula = log(unknown$Y1) ~ unknown$X1)
# Residuals:
#      Min       1Q   Median       3Q      Max 
# -0.26429 -0.02932  0.02343  0.07061  0.13519 
# Coefficients:
#              Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  6.005020   0.071561   83.92  < 2e-16 ***
# unknown$X1  -0.090368   0.007075  -12.77 2.54e-14 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Residual standard error: 0.1058 on 33 degrees of freedom
#   (4 observations deleted due to missingness)
# Multiple R-squared:  0.8317,	Adjusted R-squared:  0.8266 
# F-statistic: 163.1 on 1 and 33 DF,  p-value: 2.544e-14
  # Note: The (overall) p value is significant, which means X1 does predict (or 
    # significantly influence) Y1; reject the null and accept the alternative 
    # hypothesis
    # X1 accounts for ~83% of the variance in Y1
    # Every 1 unit increase in X1 predicts a ~0.09 unit decrease in Y1


  # Non-linear regression for a quadratic relationship
    # Create variable with IV squared
x1Sqrd = unknown$X1 ^ 2

    # Create model and view summary statistics
unknownQm1 = lm(unknown$Y1 ~ unknown$X1 + x1Sqrd)

summary(unknownQm1)
# Call:
# lm(formula = unknown$Y1 ~ unknown$X1 + x1Sqrd)
# Residuals:
#     Min      1Q  Median      3Q     Max 
# -8.1829 -3.5405  0.3129  3.3066  8.1229 
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 134.0067    10.6525   12.58 6.26e-14 ***
# unknown$X1   22.4021     2.1788   10.28 1.14e-11 ***
# x1Sqrd       -1.7723     0.1066  -16.62  < 2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Residual standard error: 4.338 on 32 degrees of freedom
#   (4 observations deleted due to missingness)
# Multiple R-squared:  0.9871,	Adjusted R-squared:  0.9863 
# F-statistic:  1221 on 2 and 32 DF,  p-value: < 2.2e-16
  # Note: The (overall) p value is significant, which means X1 does predict (or 
    # significantly influence) Y1; reject the null and accept the alternative 
    # hypothesis
    # X1 accounts for ~99% of the variance in Y1
    # Every 1 unit increase in X1 predicts a ~1.77 unit decrease in Y1
    # So - when treating this as a quadratic relationship, as decided on via
    # the graphs, there is a higher indicated explanation of variance, and thus
    # greater Y1 decrease per X1 increase, then when treating it as an 
    # exponential relationship



# Run analysis on '2' data ----
  # Note: Because these data were not distinctly quadratic, I will run both
    # types of regression

  # Non-linear regression for an exponential relationship
unknownExM2 = lm(log(unknown$Y2) ~ unknown$X2)

summary(unknownExM2)
# Call:
# lm(formula = log(unknown$Y2) ~ unknown$X2)
# Residuals:
#      Min       1Q   Median       3Q      Max 
# -0.43355 -0.07484  0.02495  0.09559  0.31863 
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  2.29060    0.05693   40.23   <2e-16 ***
# unknown$X2   0.99481    0.05189   19.17   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Residual standard error: 0.1489 on 37 degrees of freedom
# Multiple R-squared:  0.9085,	Adjusted R-squared:  0.9061 
# F-statistic: 367.6 on 1 and 37 DF,  p-value: < 2.2e-16
  # Note: The (overall) p value is significant, which means X2 does predict (or 
    # significantly influence) Y2; reject the null and accept the alternative 
    # hypothesis
    # X2 accounts for ~91% of the variance in Y2
    # Every 1 unit increase in X2 predicts a ~0.99 unit increase in Y2

  # Non-linear regression for a quadratic relationship
    # Create variable with IV squared
x2Sqrd = unknown$X2 ^ 2

    # Create model and view summary statistics
unknownQm2 = lm(unknown$Y2 ~ unknown$X2 + x1Sqrd)

summary(unknownQm2)
# Call:
# lm(formula = unknown$Y2 ~ unknown$X2 + x1Sqrd)
# Residuals:
#    Min     1Q Median     3Q    Max 
# -9.536 -3.502  0.014  2.583 13.715 
# Coefficients:
#              Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  0.481436   3.146780   0.153    0.879    
# unknown$X2  29.146697   2.175476  13.398 1.14e-14 ***
# x1Sqrd       0.001073   0.018666   0.057    0.955    
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Residual standard error: 5.703 on 32 degrees of freedom
#   (4 observations deleted due to missingness)
# Multiple R-squared:  0.8488,	Adjusted R-squared:  0.8393 
# F-statistic: 89.81 on 2 and 32 DF,  p-value: 7.477e-14
  # Note: The (overall) p value is significant, which means X2 does predict (or 
    # significantly influence) Y2; reject the null and accept the alternative 
    # hypothesis
    # X2 accounts for ~84% of the variance in Y2
    # Every 1 unit increase in X2 predicts a ~0.001 unit increase in Y2
    # So - when treating this as a exponential relationship, as decided on via
    # the graphs, there is a higher indicated explanation of variance, and thus
    # greater Y2 increase per X2 increase, then when treating it as an 
    # quadratic relationship



# Summary ----
# Both the '1' and '2' variables are significantly related
  # When the X1 variable increases, the Y1 variable decreases significantly
  # When the X2 variable increases, the Y2 variable increases significantly