# DSO106 - Machine Learning and Modeling
  # Lesson 4 - Modeling with Stepwise Regression
  # Page 8 - Modeling with Stepwise Regression Hands-On

# Requirements - Part I: Backwards Elimination ----
  # Use this data file, which contains test scores and IQ for 15 individuals. 
  # Each individual took 5 tests. The IQ is the response variable, and the five 
  # different tests are the potential predictor variables. Perform a backwards 
  # elimination on this data, then answer the following questions:
    # Which model is the best? Why?
    # From the best model, what is the adjusted R2 value and what does it mean?
    # From the best model, how does each variable influence IQ?

# Import data
iq = read.csv('/Users/hannah/Library/CloudStorage/GoogleDrive-gracesnouveaux@gmail.com/My Drive/Bethel Tech/Data Science/DSO106 Machine Learning and Modeling/1: Modeling and Optimization – Lesson 4. Modeling with Stepwise Regression/IQ.csv')

View(iq)
# 15 entries, 6 total columns

# Goal: Predict mpg
  # IV's (x axis, continuous): TBD
  # DV (y axis, continuous): IQ
  # H0: No IV's predict IQ score
    # H1: At least one IV predicts IQ score


# Create a multiple linear regression model with all IV's
iqLmAll = lm(IQ ~ ., data = iq)

summary(iqLmAll)
# Note: No potential IV's are significant on their own, though the combined 
  # effect of them together is, though all tests together only account for ~6%
  # of variance in IQ scores


  # Use backward elimination to select significant IV's
step(iqLmAll, direction = 'backward')
# ...
# Step:  AIC=71.69
# ...
# Call:
# lm(formula = IQ ~ Test1 + Test2 + Test4, data = iq)
# ...
  # Note: Kept tests 1, 2, and 4; removed tests 3 and 5


  # Run analysis based on suggestion from backward elimination
iqLmBE = lm(IQ ~ Test1 + Test2 + Test4, data = iq)

summary(iqLmBE)
# ...
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  90.7327    12.8272   7.073 2.06e-05 ***
# Test1        -1.9650     0.9406  -2.089   0.0607 .  
# Test2        -1.6485     0.7980  -2.066   0.0632 .  
# Test4         3.7890     1.6801   2.255   0.0455 *  
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Residual standard error: 9.756 on 11 degrees of freedom
# Multiple R-squared:  0.3839,	Adjusted R-squared:  0.2158 
# F-statistic: 2.284 on 3 and 11 DF,  p-value: 0.1356
  # Note: Only Test4 is significant on its own, though together with Test1 and 
    # 2 the combined effect is insignificant; because of the Test4 significance
    # reject the null and accept the alternative hypothesis
    # The combined effect of tests 1, 2, and 4 explains ~22% of the variance in 
    # IQ scores;
    # A one unit increase in Test4 score predicts a ~3.79 increase in IQ score


# Part I Summary ----
# Test 4 has a significant impact on IQ scores, with a 1 unit increase 
  # predicting a ~3.79 increase in IQ scores




# Requirements - Part II: Compare Stepwise Regression Types ----
# The following dataset will be used for this analysis. This data has a single 
  # response (Y) variable, and twelve predictor (X1 through X12) variables. Use 
  # these data to run all three kinds of stepwise regressions (backward 
  # elimination, forward selection, and the hybrid method). After completing 
  # these analyses, answer the following questions:
    # Which model was the best for each type of method?
    # How do the final models from each method compare to each other?
    # From your chosen "best model," explain what variable(s) contribute to 
      # predicting Y and for how much variance they account.


# Import data
tba = read.csv('/Users/hannah/Library/CloudStorage/GoogleDrive-gracesnouveaux@gmail.com/My Drive/Bethel Tech/Data Science/DSO106 Machine Learning and Modeling/1: Modeling and Optimization – Lesson 4. Modeling with Stepwise Regression/stepwiseRegression.csv')

View(tba)
# 128 entries, 13 total columns

# Goal: Predict Y
  # IV's (x axis, continuous): TBD
  # DV (y axis, continuous): Y
  # H0: No IV's predict Y
    # H1: At least one IV predicts Y


# Backward elimination ----

  # Create a multiple linear regression model with all IV's
tbaLmAll = lm(Y ~ ., data = tba)

summary(tbaLmAll)
# ...
# Coefficients:
#               Estimate Std. Error  t value Pr(>|t|)    
# (Intercept)  1.410e+03  1.496e+01   94.260   <2e-16 ***
# X1          -9.897e-03  1.815e-02   -0.545   0.5865    
# X2          -7.567e-02  4.301e-02   -1.759   0.0812 .  
# X3          -1.007e-01  9.998e-02   -1.007   0.3160    
# X4           2.810e+00  9.887e-03  284.252   <2e-16 ***
# X5          -1.972e-01  1.945e-01   -1.014   0.3128    
# X6           5.988e+00  7.560e-03  792.080   <2e-16 ***
# X7           1.157e-02  1.033e-02    1.120   0.2651    
# X8           2.921e-02  3.311e-02    0.882   0.3794    
# X9          -1.738e-02  3.021e-02   -0.575   0.5661    
# X10         -1.194e+01  1.886e-01  -63.302   <2e-16 ***
# X11         -1.454e-01  9.145e-02   -1.590   0.1145    
# X12         -2.601e+01  1.200e-01 -216.772   <2e-16 ***
# ...
  # Note: The combined effect is significant, as well as IV's: X4, X6, X10, X12


  # Use backward elimination to select significant IV's
step(tbaLmAll, direction = 'backward')
# ...
# Step:  AIC=213.38
# ...
# Call:
# lm(formula = Y ~ X2 + X4 + X6 + X10 + X11 + X12, data = tba)
# ...
  # Note: Kept X's 2, 4, 6, 10, 11, and 12 (and removed the rest)


  # Run analysis based on suggestion from backward elimination
tbaLmBE = lm(formula = Y ~ X2 + X4 + X6 + X10 + X11 + X12, data = tba)

summary(tbaLmBE)
# ...
# Coefficients:
#               Estimate Std. Error  t value Pr(>|t|)    
# (Intercept)  1.410e+03  1.320e+01  106.815   <2e-16 ***
# X2          -6.975e-02  4.158e-02   -1.677   0.0961 .  
# X4           2.808e+00  9.434e-03  297.644   <2e-16 ***
# X6           5.987e+00  7.410e-03  807.924   <2e-16 ***
# X10         -1.198e+01  1.742e-01  -68.769   <2e-16 ***
# X11         -1.310e-01  8.950e-02   -1.464   0.1458    
# X12         -2.598e+01  1.171e-01 -221.937   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Residual standard error: 2.241 on 121 degrees of freedom
# Multiple R-squared:  0.9999,	Adjusted R-squared:  0.9998 
# F-statistic: 1.385e+05 on 6 and 121 DF,  p-value: < 2.2e-16
  # Note: X's 4, 6, 10, and 12 are each significant, and together with 2 and 10
    # are significant and explain ~100% of the variance in Y; reject the null 
    # and accept the alternative hypothesis
    # A one unit increase in:
      # X4 predicts a 2.808e+00 unit increase in Y
      # X6 predicts a 5.987e+00 unit increase in Y
      # X10 predicts a -1.198e+01 unit decrease in Y
      # X12 predicts a -2.598e+01 unit decrease in Y



# Forward selection ----

# Create a multiple linear regression model with no IV's
tbaLmNone = lm(Y ~ 1, data = tba)

summary(tbaLmNone)
# Note: Nothing to review yet...


  # Use forward selection to select significant IV's
step(tbaLmNone, direction = 'forward', scope = formula(tbaLmAll))
# ...
# Step:  AIC=213.38
# ...
# Call:
# lm(formula = Y ~ X6 + X4 + X12 + X10 + X2 + X11, data = tba)
# ...
  # Note: Kept X's 2, 4, 6, 10, 11, and 12 (and removed the rest)
  # Same AIC and IV's selected as backward elimination! Will not re-run analysis



# Hybrid selection ----
step(tbaLmNone, direction = 'both', scope = formula(tbaLmAll))
# ...
# Step:  AIC=213.38
# ...
# Call:
# lm(formula = Y ~ X6 + X4 + X12 + X10 + X2 + X11, data = tba)
  # Note: Kept X's 2, 4, 6, 10, 11, and 12 (and removed the rest)
  # Same AIC and IV's selected as backward elimination and forward selection! 
  # Will not re-run analysis


# Summary ----
# Unexpectedly, all selection methods yielded the same results - X's 4, 6, 10, 
  # and 12 are each significant, and together with 2 and 10 are significant and 
  # explain ~100% of the variance in Y
  # A one unit increase in:
    # X4 predicts a 2.808e+00 unit increase in Y
    # X6 predicts a 5.987e+00 unit increase in Y
    # X10 predicts a -1.198e+01 unit decrease in Y
    # X12 predicts a -2.598e+01 unit decrease in Y