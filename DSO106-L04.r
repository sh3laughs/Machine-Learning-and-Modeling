# DSO106 - Machine Learning and Modeling
  # Lesson 4 - Modeling with Stepwise Regression

# Page 1 - Data Science ---- 

  # From workshop - https://vimeo.com/438421794

# Import data
happy = read.csv('/Users/hannah/Library/CloudStorage/GoogleDrive-gracesnouveaux@gmail.com/My Drive/Bethel Tech/Data Science/DSO106 Machine Learning and Modeling/1: Modeling and Optimization – Lesson 1. Modeling with Linear Regression/worldHappiness.csv')

View(happy)
# 158 entries, 12 total columns
  # Note: Though I grabbed this data from the github folder, it doesn't exactly
    # match what's in the workshop.. so hopefully that doesn't end up being
    # problematic

# Goal: Predict world happiness
  # IV's (x axis, continuous): TBD
  # DV (y axis, continuous): Happiness.Score
  # H0: No IV's predict happiness
    # H1: At least one IV predicts happiness


# NOTE: Skipping testing assumptions


# Wrangling

  # Eliminate unnecessary / unusable columns
happy1 = happy[, 4:12]

happy2 = subset(happy1, select = -Standard.Error)

View(happy2)
# 158 entries, 8 total columns
  # Note: My columns removed (or not kept) are different from hers, per 
  # different data... I'm less clear from video so far what the difference is
  # between what I kept and what she is using


# Create a multiple linear regression model with all IV's
happyLmAll = lm(Happiness.Score ~ ., data = happy2)
summary(happyLmAll)
# ...
# Economy..GDP.per.Capita.      1.000e+00  1.129e-04  8855.750   <2e-16 ***
# Family                        1.000e+00  1.153e-04  8675.863   <2e-16 ***
# Health..Life.Expectancy.      9.999e-01  1.619e-04  6175.103   <2e-16 ***
# Freedom                       9.997e-01  1.976e-04  5059.468   <2e-16 ***
# Trust..Government.Corruption. 9.999e-01  2.236e-04  4470.866   <2e-16 ***
# Generosity                    1.000e+00  2.018e-04  4956.272   <2e-16 ***
# Dystopia.Residual             1.000e+00  4.166e-05 24003.904   <2e-16 ***
# ...
  # Note: All IV's show as significant when run all together


  # Use backward elimination to select significant IV's
step(happyLmAll, direction = 'backward')
# Note: No IV's were removed


# Create a linear regression model with no IV's
happyLmNone = lm(Happiness.Score ~ 1, data = happy2)

summary(happyLmNone)
# Note: Nothing useful to review yet

  # Use forward selection to select significant IV's (manually)
step(happyLmNone, direction = 'forward', 
     scope = (~Economy..GDP.per.Capita. + Family + Health..Life.Expectancy. + 
                Freedom + Trust..Government.Corruption. + Generosity + 
                Dystopia.Residual))
# Note: No IV's were removed


  # Use hybrid selection to select significant IV's
step(happyLmNone, direction = 'both', scope = formula(happyLmAll))
# Note: No IV's were removed


# Interpret final model
happyLm = lm(formula = Happiness.Score ~ Economy..GDP.per.Capita. + 
                 Dystopia.Residual + Freedom + Family + Health..Life.Expectancy. 
               + Generosity + Trust..Government.Corruption., data = happy2)

summary(happyLm)
# Note: Confirmed that all IV's are significant, and the overall results are
  # significant; reject the null and accept the alternative hypothesis
  # 1 unit increase in:
    # Economy..GDP.per.Capita. predicts a 1.000e+00 unit increase in happiness
    # Dystopia.Residual predicts a 1.000e+00 unit increase in happiness
    # Freedom predicts a 9.997e-01 unit increase in happiness
    # Family predicts a 1.000e+00 unit increase in happiness
    # Health..Life.Expectancy. predicts a 9.999e-01 unit increase in happiness
    # Generosity predicts a 1.000e+00 unit increase in happiness
    # Trust..Government.Corruption. 9.999e-01



# Page 4 - Backward Elimination ----

  # From video - https://www.youtube.com/watch?v=0aTtMJO-pE4

# Load built-in dataset
head(mtcars)

# Goal: Predict mpg
  # IV's (x axis, continuous): TBD
  # DV (y axis, continuous): mpg
  # H0: No IV's predict mpg
    # H1: At least one IV predicts mpg


# Create a multiple linear regression model with all IV's (manually)
carsLmAll = lm(mpg ~ cyl + disp + hp + drat + wt + qsec + vs + am + gear + carb,
            data = mtcars)

summary(carsLmAll)
# Note: No variables are significant predictors, though wt is the closest with
  # a p value of 0.0633


  # Create a multiple linear regression model with all IV's (shortcut)
carsLmAll2 = lm(mpg ~ ., data = mtcars)

summary(carsLmAll2)
# Note: Same results (phew ;)


  # Use backward elimination to select significant IV's
step(carsLmAll2, direction = 'backward')
# ...
# Step:  AIC=61.31
# ...
# Call:
# lm(formula = mpg ~ wt + qsec + am, data = mtcars)
# ...
  # Note: Kept wt, qsec, and am as IV's (and removed the rest)


  # From lesson

# Goal: Predict mpg
  # IV's (x axis, continuous): TBD
  # DV (y axis, continuous): mpg
  # H0: No IV's predict mpg
    # H1: At least one IV predicts mpg


# Create a multiple linear regression model with all IV's
  # Note: Copying code from above, when we did in video
carsLmAll2 = lm(mpg ~ ., data = mtcars)

summary(carsLmAll2)
# Note: No variables are significant predictors, though wt is the closest with
  # a p value of 0.0633


  # Use backward elimination to select significant IV's
    # Note: Copying code from above, when we did in video
step(carsLmAll2, direction = 'backward')
# ...
# Step:  AIC=61.31
# ...
# Call:
# lm(formula = mpg ~ wt + qsec + am, data = mtcars)
# ...
  # Note: Kept wt, qsec, and am as IV's (and removed the rest)


  # Run analysis based on suggestion from backward elimination
carsLmBE = lm(formula = mpg ~ wt + qsec + am, data = mtcars)

summary(carsLmBE)
# ...
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)    
# (Intercept)   9.6178     6.9596   1.382 0.177915    
# wt           -3.9165     0.7112  -5.507 6.95e-06 ***
# qsec          1.2259     0.2887   4.247 0.000216 ***
# am            2.9358     1.4109   2.081 0.046716 *  
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Residual standard error: 2.459 on 28 degrees of freedom
# Multiple R-squared:  0.8497,	Adjusted R-squared:  0.8336 
# F-statistic: 52.75 on 3 and 28 DF,  p-value: 1.21e-11
  # Note: wt, qsec, and am are each significant, and together are significant,
    # together they explain ~83% of the variance in mpg; reject the null and
    # accept the alternative hypothesis
    # A one unit increase in:
      # wt predicts a 3.9165 decrease in mpg
      # qsec predicts a 1.2259 increase in mpg
      # am predicts a 2.9358 increase in mpg



# Page 5 - Forward Selection ----

  # From video - https://www.youtube.com/watch?v=OYEII--K_k4

# Create a multiple linear regression model with no IV's
carsLmNone = lm(mpg ~ 1, data = mtcars)

summary(carsLmNone)
# Note: Nothing to review yet...


  # Use forward selection to select significant IV's
step(carsLmNone, direction = 'forward', scope = formula(carsLmAll2))
# ...
# Step:  AIC=62.66
# ...
# Call:
# lm(formula = mpg ~ wt + cyl + hp, data = mtcars)
# ...
  # Note: Kept wt, cyl, and hp as IV's (and removed the rest)
  # Interesting that this is different from the backward elimination results,
  # though this AIC is higher, which means backwards was better



  # From lesson

# Create a multiple linear regression model with no IV's
  # Note: Copying code from above, when we did in video
carsLmNone = lm(mpg ~ 1, data = mtcars)

summary(carsLmNone)
# Note: Nothing to review yet...


  # Use forward selection to select significant IV's
    # Note: Copying code from above, when we did in video
step(carsLmNone, direction = 'forward', scope = formula(carsLmAll2))
# ...
# Step:  AIC=62.66
# ...
# Call:
# lm(formula = mpg ~ wt + cyl + hp, data = mtcars)
# ...
  # Note: Kept wt, cyl, and hp as IV's (and removed the rest)
  # Interesting that this is different from the backward elimination results,
  # though this AIC is higher, which means backwards was better


  # Run analysis based on suggestion from backward elimination
carsLmFS = lm(formula = mpg ~ wt + cyl + hp, data = mtcars)

summary(carsLmFS)
# ...
# Coefficients:
#             Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 38.75179    1.78686  21.687  < 2e-16 ***
# wt          -3.16697    0.74058  -4.276 0.000199 ***
# cyl         -0.94162    0.55092  -1.709 0.098480 .  
# hp          -0.01804    0.01188  -1.519 0.140015    
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Residual standard error: 2.512 on 28 degrees of freedom
# Multiple R-squared:  0.8431,	Adjusted R-squared:  0.8263 
# F-statistic: 50.17 on 3 and 28 DF,  p-value: 2.184e-11
  # Note: wt is significant on its own, and the combined effect of wt, cyl, and
    # hp together is significant, together they explain ~83% of the variance in 
    # mpg; reject the null and accept the alternative hypothesis
    # A one unit increase in wt predicts a 3.16697 decrease in mpg
    # The overall effect of these variables explains the same % of change in
    # mpg as the variables from the backward elimination, the difference in
    # impact from weight was similar, and cyl and hp were not significant on 
    # their own with either method of selection



# Page 6 - Hybrid Stepwise Regression ----

  # From video - https://www.youtube.com/watch?v=ejR8LnQziPY

# Use hybrid selection to select significant IV's
step(carsLmNone, direction = 'both', scope = formula(carsLmAll2))
# ...
# Step:  AIC=62.66
# ...
# Call:
# lm(formula = mpg ~ wt + cyl + hp, data = mtcars)
# ...
  # Note: Kept wt, cyl, and hp as IV's (and removed the rest)
  # This is identical results to the forward selection option



  # From lesson

# Use hybrid selection to select significant IV's
    # Note: Copying code from above, when we did in video
step(carsLmNone, direction = 'both', scope = formula(carsLmAll2))
# ...
# Step:  AIC=62.66
# ...
# Call:
# lm(formula = mpg ~ wt + cyl + hp, data = mtcars)
# ...
  # Note: Kept wt, cyl, and hp as IV's (and removed the rest)
  # This is identical results to the forward selection option
