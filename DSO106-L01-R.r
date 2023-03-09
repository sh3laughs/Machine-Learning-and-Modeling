# DSO106 - Machine Learning and Modeling
  # Lesson 1 - Modeling with Linear Regression

# Page 6 - Regression Setup in R ----

  # From workshop: https://vimeo.com/463655964 ----

# Install and import packages
install.packages('caret')
install.packages('e1071')
install.packages('gvlma')
install.packages('lmtest')
install.packages('predictmeans')

library(car)
library(caret)
library(e1071)
library(gvlma)
library(lmtest)
library(predictmeans)

# Import and preview data
worldHappiness = read.csv('/Users/hannah/Library/CloudStorage/GoogleDrive-gracesnouveaux@gmail.com/My Drive/Bethel Tech/Data Science/DSO106 Machine Learning and Modeling/1: Modeling and Optimization – Lesson 1. Modeling with Linear Regression/worldHappiness.csv')

View(worldHappiness)


  # Goal: Predict world happiness
    # IV's (x axis, continuous):
      # Economy..GDP.per.Capita.
      # Generosity
    # DV (y axis, continuous): Happiness.Score
    # H0: Economy, generosity, or their combined effect do not predict 
        # happiness (aka: b1 = 0)
      # H1: Economy, generosity, or their combined effect do predict happiness
        # (aka: b1 ≠ 0)


# Test assumptions

  # Linearity for each potential IV
scatter.smooth(x = worldHappiness$Happiness.Score, y = worldHappiness$Family)
# Note: This is decently linear, but we're not using

scatter.smooth(x = worldHappiness$Happiness.Score, y = worldHappiness$Freedom)
# Note: Not linear, can't use

scatter.smooth(x = worldHappiness$Happiness.Score, y = 
                 worldHappiness$Economy..GDP.per.Capita.)
# Note: This is decently linear, will use

scatter.smooth(x = worldHappiness$Happiness.Score, y = 
                 worldHappiness$Trust..Government.Corruption.)
# Note: Not linear, can't use

scatter.smooth(x = worldHappiness$Happiness.Score, y = 
                 worldHappiness$Health..Life.Expectancy.)
# Note: This is decently linear, but we're not using

scatter.smooth(x = worldHappiness$Happiness.Score, y = 
                 worldHappiness$Generosity)
# Note: Not linear, but we're using

scatter.smooth(x = worldHappiness$Happiness.Score, y = 
                 worldHappiness$Dystopia.Residual)
# Note: This is decently linear, but we're not using


  # Homoscedasticity
wrldHapLm = lm(Happiness.Score ~ Economy..GDP.per.Capita. + Generosity, 
               data = worldHappiness)

    # Option 1: graphs
par(mfrow = c(2, 2))
plot(wrldHapLm)
# Note: Looking at graphs on left + bottom right, we have decent enough
  # linearity to validate this assumption

    # Option 2: Breusch Pagan Homoscedasticity test
lmtest::bptest(wrldHapLm)
# 	studentized Breusch-Pagan test
# 
# data:  wrldHapLm
# BP = 0.30467, df = 2, p-value = 0.8587
  # Note: This p value above 0.05 validates this assumption


  # Homogeneity of variance
gvlma(wrldHapLm)
# Call:
# lm(formula = Happiness.Score ~ Economy..GDP.per.Capita. + Generosity, 
#     data = worldHappiness)
# Coefficients:
#              (Intercept)  Economy..GDP.per.Capita.                Generosity  
#                    3.090                     2.224                     1.704  
# ASSESSMENT OF THE LINEAR MODEL ASSUMPTIONS
# USING THE GLOBAL TEST ON 4 DEGREES-OF-FREEDOM:
# Level of Significance =  0.05 
# Call:
#  gvlma(x = wrldHapLm) 
#                     Value p-value                Decision
# Global Stat        3.8091 0.43246 Assumptions acceptable.
# Skewness           0.5460 0.45994 Assumptions acceptable.
# Kurtosis           0.1962 0.65779 Assumptions acceptable.
# Link Function      2.7563 0.09687 Assumptions acceptable.
# Heteroscedasticity 0.3106 0.57732 Assumptions acceptable.
  # Note: Assumption validated


  # Outliers

    # Leverage outliers (x axis) - option 1
CookD(wrldHapLm, group = NULL, idn = 3, newwd = FALSE)
# Note: Outliers on rows 129, 143, 156

    # Leverage outliers (x axis) - option 2
wldHapLev = hat(model.matrix(wrldHapLm))
plot(wldHapLev)
# Note: There appears to be an outlier, but it's not labeled; no values appear
  # to be over 0.2 

    # Distance outliers (y axis) - option 1
car::outlierTest(wrldHapLm)
# No Studentized residuals with Bonferroni p < 0.05
# Largest |rstudent|:
#      rstudent unadjusted p-value Bonferroni p
# 156 -3.629181         0.00038632     0.061038
  # Note: Outlier on row 156 (which implies it is an influential outlier [x and
    # y axes], per being found in both Leverage [x] and Distance [y] tests)
    # The -3.6... value is saying that row 156 is the only row with that high
    # of a standard deviation away from the mean

    # Influence outliers (x and y axes)
summary(influence.measures(wrldHapLm))
# Potentially influential observations of
# 	 lm(formula = Happiness.Score ~ Economy..GDP.per.Capita. + Generosity,      
        # 	 data = worldHappiness) :
#     dfb.1_ dfb.E..G dfb.Gnrs dffit   cov.r   cook.d hat    
# 12   0.01   0.05     0.03     0.20    0.92_*  0.01   0.01  
# 14   0.10   0.08    -0.14     0.25    0.93_*  0.02   0.01  
# 16   0.10   0.06    -0.13     0.22    0.94_*  0.02   0.01  
# 21  -0.02   0.01     0.02     0.02    1.07_*  0.00   0.05  
# 34  -0.04   0.01     0.08     0.08    1.07_*  0.00   0.05  
# 37   0.07  -0.04    -0.10    -0.11    1.06_*  0.00   0.04  
# 119  0.00  -0.01     0.01     0.01    1.06_*  0.00   0.04  
# 129  0.15   0.14    -0.44    -0.48_*  1.16_*  0.08   0.14_*
# 143 -0.16  -0.10     0.27    -0.35    0.92_*  0.04   0.02  
# 156  0.15   0.13    -0.54    -0.63_*  0.82_*  0.12   0.03 
  # Note: Looking for values > 2 in the df columns, and none exist, which 
    # implies there are not actually any influential outliers


# Running analysis
  # Note: I went ahead and created new datasets w/o outliers, though video
    # skipped this for the sake of time

  # Step 1: Create second dataset without most significant outlier, and model
    # for that dataset (less rigorous)
worldHappiness2 = worldHappiness[-156,]

View(worldHappiness2)
# 157 entries, 12 total columns

wrldHapLm2 = lm(Happiness.Score ~ Economy..GDP.per.Capita. + Generosity, 
                data = worldHappiness2)


  # Step 2: Create second dataset without any outliers (more rigorous)
worldHappiness3 = worldHappiness[-c(129,143,156),]

View(worldHappiness3)
# 155 entries, 12 total columns

wrldHapLm3 = lm(Happiness.Score ~ Economy..GDP.per.Capita. + Generosity, 
                data = worldHappiness3)


  # Step 3: Get summary of original model
summary(wrldHapLm)
# Call:
# lm(formula = Happiness.Score ~ Economy..GDP.per.Capita. + Generosity, 
#     data = worldHappiness)
# Residuals:
#      Min       1Q   Median       3Q      Max 
# -2.36245 -0.43946 -0.01671  0.54241  1.58794 
# Coefficients:
#                          Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                3.0898     0.1642  18.816  < 2e-16 ***
# Economy..GDP.per.Capita.   2.2238     0.1359  16.369  < 2e-16 ***
# Generosity                 1.7038     0.4323   3.941 0.000122 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Residual standard error: 0.6862 on 155 degrees of freedom
# Multiple R-squared:  0.6454,	Adjusted R-squared:  0.6409 
# F-statistic: 141.1 on 2 and 155 DF,  p-value: < 2.2e-16
  # Note: Overall p value (bottom line) is significant, so we can continue with
    # interpretation
    # Each IV is also significant
      # $1 (?) increase in per capita GDP predicts a ~2.22 increase in happiness
      # 1 unit increase in generosity predicts a ~1.7 increase in happiness
    # Looking at Adjusted R-squared value: Together, these IV's account for 
      # ~64% of the variance in world happiness


  # Step 4: Get summary of new, less rigorous model w/o significant outlier
summary(wrldHapLm2)
# Call:
# lm(formula = Happiness.Score ~ Economy..GDP.per.Capita. + Generosity, 
#     data = worldHappiness2)
# Residuals:
#      Min       1Q   Median       3Q      Max 
# -1.64141 -0.44310 -0.03168  0.51687  1.59726 
# Coefficients:
#                          Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                3.0657     0.1583  19.372  < 2e-16 ***
# Economy..GDP.per.Capita.   2.2071     0.1309  16.861  < 2e-16 ***
# Generosity                 1.9298     0.4209   4.585 9.34e-06 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Residual standard error: 0.6607 on 154 degrees of freedom
# Multiple R-squared:  0.6642,	Adjusted R-squared:  0.6598 
# F-statistic: 152.3 on 2 and 154 DF,  p-value: < 2.2e-16
  # Note: Though the values differ slightly, the results are the same as the
    # original data
    # The biggest difference is in the increased significance of Generosity as 
    # a predictor, which implies the outlier was an unusually low value for 
    # this variable
    # The other difference is in the Adjusted R-Squared - without this outlier, 
    # these variables have an even greater ability to predict happiness

  # Step 5: Get summary of new, less rigorous model w/o significant outlier
summary(wrldHapLm3)
# Call:
# lm(formula = Happiness.Score ~ Economy..GDP.per.Capita. + Generosity, 
#     data = worldHappiness3)
# Residuals:
#      Min       1Q   Median       3Q      Max 
# -1.46182 -0.46435 -0.04755  0.49809  1.59252 
# Coefficients:
#                          Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                3.0633     0.1567  19.554  < 2e-16 ***
# Economy..GDP.per.Capita.   2.1989     0.1293  17.006  < 2e-16 ***
# Generosity                 2.0401     0.4448   4.587 9.34e-06 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Residual standard error: 0.6472 on 152 degrees of freedom
# Multiple R-squared:  0.6764,	Adjusted R-squared:  0.6721 
# F-statistic: 158.8 on 2 and 152 DF,  p-value: < 2.2e-16
  # Note: Though the values differ slightly, the results are the same as the
    # original data
    # The biggest difference is in the increased significance of Generosity as 
    # a predictor, which implies the outlier was an unusually low value for 
    # this variable (NOTE: there is no difference in the Generosity 
    # significance from the other dataset which removed an outlier, which 
    # implies the other 2 outliers were less significant)
    # The other difference is in the Adjusted R-Squared - without outliers, 
    # these variables have an even greater ability to predict happiness (than
    # with outliers, or even just without the single outlier removed above)



  # From lesson ----

# Import and preview data
manatees = read.csv('/Users/hannah/Library/CloudStorage/GoogleDrive-gracesnouveaux@gmail.com/My Drive/Bethel Tech/Data Science/DSO106 Machine Learning and Modeling/1: Modeling and Optimization – Lesson 1. Modeling with Linear Regression/manatees.csv')

View(manatees)
# 35 entries, 2 total columns


# Goal: Determine whether the number of powerboats registered in Florida 
    # impacts the number of manatees killed there by powerboats each year
  # IV (x axis, continuous): powerboats
  # DV (y axis, continuous): manatee deaths
  # H0: The number of powerboats registered in FL does not predict the number
      # of manatees killed there by powerboats each year (aka: b1 = 0)
    # H1: The number of powerboats registered in FL does predict the number of
      # manatees killed there by powerboats each year (aka: b1 ≠ 0)


# Test Assumptions

  # Linearity
scatter.smooth(x = manatees$PowerBoats, y = manatees$ManateeDeaths, 
               main = 'Manatee Deaths by Power Boats')
# Note: Data are linear enough ;)


  # Homoscedasticity

    # Create linear model
manateesLm = lm(ManateeDeaths ~ PowerBoats, data = manatees)

    # Option 1: Graph model
par(mfrow = c(2, 2))
plot(manateesLm)
# Note: This appears to have some non-linear results, which means the 
  # assumption is most likely violated and the data are heteroscedastic


    # Option 2: Breusch Pagan Homoscedasticity test
lmtest::bptest(manateesLm)
# 	studentized Breusch-Pagan test
# data:  manateesLm
# BP = 7.8867, df = 1, p-value = 0.00498
  # Note: This p value below 0.05 violates the assumption (confirming graph
    # results above)


    # Option 3: Non-constant Variance Score (NCV) test
car::ncvTest(manateesLm)
# Non-constant Variance Score Test 
# Variance formula: ~ fitted.values 
# Chisquare = 9.037421, Df = 1, p = 0.0026451
  # Note: This p value below 0.05 violates the assumption (confirming results 
    # above)


    # Correct for heteroscadasticity
manateesBc = caret::BoxCoxTrans(manatees$ManateeDeaths)

print(manateesBc)
# Box-Cox Transformation
# 35 data points used to estimate Lambda
# Input data summary:
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#    14.0    30.5    50.0    52.8    75.5    97.0 
# Largest/Smallest: 6.93 
# Sample Skewness: 0.173 
# Estimated Lambda: 0.5
  # Note: Printing results was just for practice.. not actually necessary

manatees = cbind(manatees, newDv = predict(manateesBc, manatees$ManateeDeaths))

manateesLm2 = lm(newDv ~ PowerBoats, data = manatees)

lmtest::bptest(manateesLm2)
# 	studentized Breusch-Pagan test
# data:  manateesLm2
# BP = 5.6078, df = 1, p-value = 0.01788
  # Note: This p value below 0.05 still violates the assumption, but we're
    # proceeding anyway


  # Homogeneity of variance
gvlma(manateesLm2)
# Call:
# lm(formula = newDv ~ PowerBoats, data = manatees)
# Coefficients:
# (Intercept)   PowerBoats  
#    -1.76474      0.01872  
# ASSESSMENT OF THE LINEAR MODEL ASSUMPTIONS
# USING THE GLOBAL TEST ON 4 DEGREES-OF-FREEDOM:
# Level of Significance =  0.05 
# Call:
#  gvlma(x = manateesLm2) 
#                      Value p-value                   Decision
# Global Stat        12.1803 0.01606 Assumptions NOT satisfied!
# Skewness            1.1918 0.27496    Assumptions acceptable.
# Kurtosis            0.1569 0.69207    Assumptions acceptable.
# Link Function       6.9611 0.00833 Assumptions NOT satisfied!
# Heteroscedasticity  3.8705 0.04914 Assumptions NOT satisfied!
  # Note: The assumption is violated, per the violation note for 
    # heteroscedasticity here (according to chatbotGPT, homoscedasticity and
    # homogeneity of variance are the same thing... so I'm not sure why we're
    # being taught that they are different assumptions, given that this test
    # aligns with that concept...)


  # Leverage outliers
CookD(manateesLm2, group = NULL, idn = 3, newwd = FALSE)
# Note: Outliers on rows 24, 29, 31


    # Confirm leverage values
manateesLev = hat(model.matrix(manateesLm2))

plot(manateesLev)
# Note: No values appear to be above 0.2, which means no significant leverage 
  # outliers after all...


    # Confirm no leverage values are above 0.2
manatees[manateesLev > .2,]
# [1] PowerBoats    ManateeDeaths newDv        
# <0 rows> (or 0-length row.names)
  # Note: Confirmed!


  # Distance outliers
car::outlierTest(manateesLm2)
# No Studentized residuals with Bonferroni p < 0.05
# Largest |rstudent|:
#    rstudent unadjusted p-value Bonferroni p
# 23 2.446086           0.020121      0.70423
  # Note: This p value above 0.05 confirms there are no distance outliers


  # Influential outliers
summary(influence.measures(manateesLm2))
# Potentially influential observations of
# 	 lm(formula = newDv ~ PowerBoats, data = manatees) :
#    dfb.1_ dfb.PwrB dffit cov.r   cook.d hat  
# 23 -0.07   0.18     0.46  0.78_*  0.09   0.03
# 24 -0.18   0.29     0.50  0.80_*  0.11   0.04
  # Note: The dfb.1_ and dffit values all being below 1 indicates there are no
    # influential outliers


# Page 7 - Regression in R ----

# Run analysis

  # Transformed data
summary(manateesLm2)
# Call:
# lm(formula = newDv ~ PowerBoats, data = manatees)
# Residuals:
#      Min       1Q   Median       3Q      Max 
# -2.11978 -0.95827 -0.05081  0.75875  2.74620 
# Coefficients:
#              Estimate Std. Error t value Pr(>|t|)    
# (Intercept) -1.764745   0.842078  -2.096   0.0439 *  
# PowerBoats   0.018718   0.001106  16.929   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Residual standard error: 1.226 on 33 degrees of freedom
# Multiple R-squared:  0.8967,	Adjusted R-squared:  0.8936 
# F-statistic: 286.6 on 1 and 33 DF,  p-value: < 2.2e-16
  # Note: This p value indicates the test is significant - reject the null and
    # accept the alternative hypothesis; this means that the number of 
    # powerboats registered in FL does predict the number of manatees killed 
    # there by powerboats each year, it explains ~90% of the variance


  # Original data
summary(manateesLm)
# Call:
# lm(formula = ManateeDeaths ~ PowerBoats, data = manatees)
# Residuals:
#     Min      1Q  Median      3Q     Max 
# -15.736  -6.642  -1.239   4.374  22.309 
# Coefficients:
#               Estimate Std. Error t value Pr(>|t|)    
# (Intercept) -42.525657   6.347072    -6.7 1.25e-07 ***
# PowerBoats    0.129133   0.008334    15.5  < 2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Residual standard error: 9.237 on 33 degrees of freedom
# Multiple R-squared:  0.8792,	Adjusted R-squared:  0.8755 
# F-statistic: 240.1 on 1 and 33 DF,  p-value: < 2.2e-16
  # Note: The original data yields the same results as the transformed data;
    # the main difference is that the original data explains slightly less 
    # (~88%) variance than the transformed data (~90%)