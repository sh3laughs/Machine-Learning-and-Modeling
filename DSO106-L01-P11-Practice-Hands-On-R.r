# DSO106 - Machine Learning and Modeling
  # Lesson 1 - Modeling with Linear Regression
  # Page 11 - Best Fit Line Hands-On

# Setup ----

# Requirements: It is a well known phenomena that most of us shrink throughout 
    # the day each day. The effects of gravity cause that our height measured 
    # at the end of the day is less than our height measured at the beginning 
    # of the day. Fortunately, at night, our bodies stretch out again, so that 
    # from one morning to the next, each of us has returned to the morning 
    # height from the day before.
  # In the dataset below, there are AM and PM height measurements (in mm) for 
    # students from a boarding school in India.
  # Hands on Part 1: Take the following dataset, and complete simple linear 
    # regression in R. Make sure to test, note, and correct for all assumptions 
    # if possible!


# Install and import packages
library(car)
library(caret)
library(e1071)
library(gvlma)
library(lmtest)
library(predictmeans)

# Import and preview data
height = read.csv('/Users/hannah/Library/CloudStorage/GoogleDrive-gracesnouveaux@gmail.com/My Drive/Bethel Tech/Data Science/DSO106 Machine Learning and Modeling/1: Modeling and Optimization – Lesson 1. Modeling with Linear Regression/heights.csv')

View(height)
# 41 entries, 2 total columns


# Goal: Predict height
  # IV (x axis, continuous): AM_Height
  # DV (y axis, continuous): PM_Height
  # H0: Morning height does not predict evening height (aka: b1 = 0)
    # H1: Morning height does predict evening height (aka: b1 ≠ 0)

# Create linear regression model
heightLm = lm(PM_Height ~ AM_Height, data = height)



# Test assumptions ----

  # Linearity
scatter.smooth(x = height$AM_Height, y = height$PM_Height)
# Note: Super linear! Assumption validated


  # Homoscedasticity
lmtest::bptest(heightLm)
# 	studentized Breusch-Pagan test
# data:  heightLm
# BP = 0.2079, df = 1, p-value = 0.6484
  # Note: This p value is insignificant, which validates this assumption


  # Homogeneity of variance
gvlma(heightLm)
# Call:
# lm(formula = PM_Height ~ AM_Height, data = height)
# Coefficients:
# (Intercept)    AM_Height  
#      8.6537       0.9891  
# ASSESSMENT OF THE LINEAR MODEL ASSUMPTIONS
# USING THE GLOBAL TEST ON 4 DEGREES-OF-FREEDOM:
# Level of Significance =  0.05 
# Call:
#  gvlma(x = heightLm) 
#                     Value p-value                Decision
# Global Stat        2.6500  0.6180 Assumptions acceptable.
# Skewness           0.0946  0.7584 Assumptions acceptable.
# Kurtosis           0.4953  0.4816 Assumptions acceptable.
# Link Function      1.8925  0.1689 Assumptions acceptable.
# Heteroscedasticity 0.1676  0.6823 Assumptions acceptable.
  # Note: All assumptions validated (as made believable by results in test
    # above)
    # Will be asking on a code review why we test for both homoscedasticity AND
    # homogeneity of variance...


  # Leverage outliers
CookD(heightLm, group = NULL, idn = 3, newwd = FALSE)
# Note: Outliers on rows 3, 4, 12

    # Confirm leverage values
heightLev = hat(model.matrix(heightLm))

plot(heightLev)
# Note: One value is just baredly above 0.2...

    # Confirm whether there are leverage values above 0.2
height[heightLev > .2,]
#   AM_Height PM_Height
# 3   1462.25    1452.5
  # Note: Row 3 truly is a leverage outlier


  # Distance outliers
car::outlierTest(heightLm)
# No Studentized residuals with Bonferroni p < 0.05
# Largest |rstudent|:
#     rstudent unadjusted p-value Bonferroni p
# 37 -2.263445           0.029403           NA
  # Note: This p value below 0.05 confirms that (row 37 ?) is a distance outlier


  # Influential outliers
summary(influence.measures(heightLm))
# Potentially influential observations of
# 	 lm(formula = PM_Height ~ AM_Height, data = height) :
#    dfb.1_ dfb.AM_H dffit cov.r   cook.d hat    
# 3  -0.54   0.53    -0.56  1.26_*  0.16   0.21_*
# 11  0.01  -0.01    -0.01  1.18_*  0.00   0.10  
# 37 -0.19   0.17    -0.40  0.84_*  0.07   0.03
  # Note: This confirms row 3 is actually an influential outlier (not sure why
    # it didn't show up in the distance test...), rows 11 and 37 are included,
    # though not significant...


    # Create dataset without most significant outlier (less rigorous), and its
      # model
height2 = height[-3,]

View(height2)
# 40 entries, 2 total columns

heightLm2 = lm(PM_Height ~ AM_Height, data = height2)


    # Create dataset without any outliers (more rigorous), and its model
height3 = height[-c(3, 4, 11, 12, 37),]

View(height3)
# 36 entries, 2 total columns

heightLm3 = lm(PM_Height ~ AM_Height, data = height3)



# Run analysis ----

  # Original model
summary(heightLm)
# Call:
# lm(formula = PM_Height ~ AM_Height, data = height)
# Residuals:
#     Min      1Q  Median      3Q     Max 
# -5.5694 -1.9884 -0.1255  1.6838  5.4790 
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 8.653685   8.714042   0.993    0.327    
# AM_Height   0.989149   0.005177 191.066   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Residual standard error: 2.627 on 39 degrees of freedom
# Multiple R-squared:  0.9989,	Adjusted R-squared:  0.9989 
# F-statistic: 3.651e+04 on 1 and 39 DF,  p-value: < 2.2e-16
  # Note: The overall p value is significant; reject the null and accept the 
    # alternative hypothesis - morning height does predict night height (that's
    # a rhyme!)
    # According to the adjusted R value, morning height explains ~100% of the
    # variance in evening height! 


  # Less rigorous, one significant outlier removed model
summary(heightLm2)
# Call:
# lm(formula = PM_Height ~ AM_Height, data = height2)
# Residuals:
#    Min     1Q Median     3Q    Max 
# -5.751 -2.099 -0.039  1.700  5.372 
# Coefficients:
#              Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 13.330012   9.695373   1.375    0.177    
# AM_Height    0.986414   0.005742 171.777   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Residual standard error: 2.621 on 38 degrees of freedom
# Multiple R-squared:  0.9987,	Adjusted R-squared:  0.9987 
# F-statistic: 2.951e+04 on 1 and 38 DF,  p-value: < 2.2e-16
  # Note: Only a very tiny difference from original model


  # More rigorous, all possible outliers removed model
summary(heightLm3)
# Call:
# lm(formula = PM_Height ~ AM_Height, data = height3)
# Residuals:
#    Min     1Q Median     3Q    Max 
# -5.101 -1.743 -0.189  1.678  5.246 
# Coefficients:
#              Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 16.835236   9.896369   1.701    0.098 .  
# AM_Height    0.984392   0.005892 167.079   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Residual standard error: 2.374 on 34 degrees of freedom
# Multiple R-squared:  0.9988,	Adjusted R-squared:  0.9987 
# F-statistic: 2.792e+04 on 1 and 34 DF,  p-value: < 2.2e-16
  # Note: Also only a very tiny difference from original model



# Summary ----
# It is safe to say that morning height accounts for all variance in predicting
  # evening height