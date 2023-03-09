# DSO106 - Machine Learning and Modeling
  # Lesson 3 - Non-Linear Modeling

# Page 3 - Quadratic Modeling in R ----

# Import packages
library(ggplot2)


# Import data
fish = read.csv('/Users/hannah/Library/CloudStorage/GoogleDrive-gracesnouveaux@gmail.com/My Drive/Bethel Tech/Data Science/DSO106 Machine Learning and Modeling/1: Modeling and Optimization – Lesson 3. Non-Linear Modeling/bluegill_fish.csv')

View(fish)
# 78 entries, 2 total columns


# Goal: Determine whether the age of bluegill fish influences their length
  # IV (x axis, continuous): age
  # DV (y axis, categorical): length
  # H0: The age of bluegill fish does not influence their length
    # H1: The age of bluegill fish does influence their length


# Test assumption - quadratic relationship
ggplot(fish, aes(x = age, y = length)) +
  geom_point() + 
  stat_smooth(method = 'lm', formula = y ~ x + I(x ^ 2), size = 1)
# Note: Validated!


# Run analysis

  # Create variable with IV squared
ageSqrd = fish$age ^ 2

  # Create model and view summary statistics
fishQm = lm(fish$length ~ fish$age + ageSqrd)

summary(fishQm)
# Call:
# lm(formula = fish$length ~ fish$age + ageSqrd)
# Residuals:
#      Min       1Q   Median       3Q      Max 
# -18.6170  -5.7699  -0.6662   5.6881  18.1085 
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)    
# (Intercept)   2.4242     9.5976   0.253    0.801    
# fish$age     50.4923     5.2141   9.684 7.53e-15 ***
# ageSqrd      -3.6511     0.6951  -5.253 1.36e-06 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Residual standard error: 7.911 on 75 degrees of freedom
# Multiple R-squared:  0.8954,	Adjusted R-squared:  0.8926 
# F-statistic: 320.9 on 2 and 75 DF,  p-value: < 2.2e-16
  # Note: The (overall) p value is significant, which means age does predict
    # length; reject the null and accept the alternative hypothesis
    # Age accounts for ~89% of the variance in length, and every 1 unit increase
    # in age predicts a ~50.49 unit increase in length



# Page 6 - Exponential Modeling in 5 ----

# Import data
bacteria = read.csv('/Users/hannah/Library/CloudStorage/GoogleDrive-gracesnouveaux@gmail.com/My Drive/Bethel Tech/Data Science/DSO106 Machine Learning and Modeling/1: Modeling and Optimization – Lesson 3. Non-Linear Modeling/bacteria.csv')

View(bacteria)
# 15 entries, 2 total columns


# Goal: Determine how much does bacteria grows over time / whether time predicts
    # growth
  # IV (x axis, continuous): time
  # DV (y axis, categorical): growth
  # H0: Time does not affect / predict bacteria growth
    # H1: Time does affect / predict bacteria growth


# Test assumption - exponential relationship
  # Note: Not covered in lesson :(


# Run analysis
bacteriaExM = lm(log(bacteria$Count) ~ bacteria$Period)

summary(bacteriaExM)
# Call:
# lm(formula = log(bacteria$Count) ~ bacteria$Period)
# Residuals:
#       Min        1Q    Median        3Q       Max 
# -0.106253 -0.038496  0.006744  0.029930  0.077803 
# Coefficients:
#                 Estimate Std. Error t value Pr(>|t|)    
# (Intercept)     2.701306   0.027086   99.73  < 2e-16 ***
# bacteria$Period 0.165330   0.003293   50.21 2.84e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Residual standard error: 0.0551 on 13 degrees of freedom
# Multiple R-squared:  0.9949,	Adjusted R-squared:  0.9945 
# F-statistic:  2521 on 1 and 13 DF,  p-value: 2.841e-16
  # Note: The (overall) p value is significant, which means time does predict
    # bacteria growth; reject the null and accept the alternative hypothesis
    # Time accounts for ~99% of the variance in growth, and every 1 unit 
    # increase in time predicts a ~.17 unit increase in growth