# DSO106 - Machine Learning and Modeling
  # Lesson 2 - Modeling with Logistic Regression

# Page 1 - Data Science ----

  # From workshop: https://vimeo.com/465050172

# Install and import packages
install.packages('magrittr')
install.packages('popbio')

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
instruments = read.csv('/Users/hannah/Library/CloudStorage/GoogleDrive-gracesnouveaux@gmail.com/My Drive/Bethel Tech/Data Science/DSO106 Machine Learning and Modeling/1: Modeling and Optimization – Lesson 2. Modeling with Logistic Regression/Musical-instruments-reviews.csv')

View(instruments)
# 10,261 entries, 9 total columns
  # Note: In video her starting dataset has 1,093 rows, 9 total columns, so
    # there must have been wrangling she did on the data in the Github folder
    # (where I pulled this from) before the workshop... we'll see how that
    # impacts my ability to play along!


# Goal: Determine whether overall rankings predict helpful musical instrument 
    # reviews
  # IV (x axis, continuous): overall
  # DV (y axis, categorical): helpful
  # H0: Overall ranking of a musical instrument review does not predict how 
      # helpful the review is 
    # H1: Overall ranking of a musical instrument review does predict how 
      # helpful the review is


# Wrangling

  # Recode outcome variable into 0's and 1's
instruments$helpfulYn = NA
instruments$helpfulYn[instruments$helpful == '[0, 0]'] = 0
instruments$helpfulYn[instruments$helpful == '[1, 1]'] = 1

View(instruments)
# 10,261 entries, 10 total columns
  # Note: There are some NA's retained from data that wasn't recoded


  # Remove missing values
reviews = NaRV.omit(instruments)

View(reviews)
# 7,955 entries, 10 total columns


# Create logistic regression model
reviewsLog = glm(helpfulYn ~ overall, data = reviews, family = binomial)


# Test assumptions

  # Sample size

    # Create probabilities (prediction) column
reviewsProb = predict(reviewsLog, type = 'response')
reviews$predict = ifelse(reviewsProb > 0.05, 'pos', 'neg')

View(reviewsProb)
# 7,955 entries, 11 total columns
  # Note: These are all positive... the first indication that the data I had
    # access to won't work, but we'll see...


    # Recode this new column
reviews$predictR = NA
reviews$predictR[reviews$predict == 'pos'] = 1
reviews$predictR[reviews$predict == 'neg'] = 0

View(reviews)
# 7,955 entries, 12 total columns
  # Note: Not surprisingly, column now has all 1's...
    # Also not sure why we did this way vs. coding as 0 and 1 instead of neg
    # and pos... UPDATE: in lesson content I learned that this was so that we
    # had both text values (for human readability) and #'s (for these tests)

    # Convert to factor (aka: categorical) data type
reviews$predictR = as.factor(reviews$predictR)
reviews$helpfulYn = as.factor(reviews$helpfulYn)

str(reviews)
# Note: Success! 
  # Though this also confirms that the predict variable only has 1 level... I
  # have a feeling my ability to keep up is only going downhill from here ;)


    # Create a confusion matrix
reviewsCm = caret::confusionMatrix(reviews$predictR, reviews$helpfulYn)

reviewsCm
# Confusion Matrix and Statistics
#           Reference
# Prediction    0    1
#          0    0    0
#          1 6796 1159
#                Accuracy : 0.1457
# ...
  # Note: Because of having 2 cells with 0's, the assumption is violated, IRL we
    # should stop here, but for practice we'll keep going...


  # Logit linearity

    # Subset data
reviews1 = reviews %>% dplyr::select_if(is.numeric)

View(reviews1)
# 7,955 entries, 2 total columns


    # Save column names to a new variable
reviewsPredict = colnames(reviews1)

    # Create new dataframe with additional info
reviews2 = reviews1 %>% 
  mutate(reviewsLogit = log(reviewsProb / 1 - reviewsProb)) %>% 
    # Creates logit
  gather(key = 'reviewsPredict', value = 'reviewsPredict.value', 
         -reviewsLogit)
    # Reshapes data


    # Plot the new dataframe
ggplot(reviews2, aes(reviewsLogit, reviewsPredict.value)) +
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = 'loess') +
  theme_bw() +
  facet_wrap(~reviewsPredict, scales = 'free_y')
# Note: Blank white square, haha
  # Warning: Removed 15910 rows containing non-finite values - again, not
  # surprising since my data is wrong, but I'll keep going for practice
  

  # Multicollinearity - skipped per having only 1 IV


  # Independent errors

    # Plot
plot(reviewsLog$residuals)
# Note: My plot looks nothing like the video ;) but I march on!


    # Durbin-Watson test
dwtest(reviewsLog, alternative = 'two.sided')
# 	Durbin-Watson test
# data:  reviewsLog
# DW = 1.7766, p-value < 2.2e-16
# alternative hypothesis: true autocorrelation is not 0
  # Note: My p value doesn't match the video, of course, but, like hers, b/c 
    # it's significant the assumption is probably violated
    # BUT you then also look at the DW value: 1.7766 - this needs to be btw.
    # 1-3, which this is, so we did actually validate the assumption. Phew!


  # Outliers

    # Influential
reviewsInfl = influence.measures(reviewsLog)
summary(reviewsInfl)
# Note: Video skipped analyzing this - just said there are outliers and we're
  # ignoring...
  # Given the difference in my data and hers, I actually don't think I have
  # outliers... but am not taking the time to fully check since my data is
  # a mess anyway


# Examine output of model
summary(reviewsLog)
# Call:
# glm(formula = helpfulYn ~ overall, family = binomial, data = reviews)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -0.6470  -0.5513  -0.5513  -0.5513   1.9794  
# Coefficients:
#             Estimate Std. Error z value Pr(>|z|)    
# (Intercept) -1.36997    0.18029  -7.599 2.99e-14 ***
# overall     -0.08743    0.03907  -2.238   0.0252 *  
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# (Dispersion parameter for binomial family taken to be 1)
# 
#     Null deviance: 6605.3  on 7954  degrees of freedom
# Residual deviance: 6600.5  on 7953  degrees of freedom
# AIC: 6604.5
# Number of Fisher Scoring iterations: 4
  # Note: Well... her results are not significant, but mine are... unreliable 
    # either way since assumptions were violated

  # Graph the model
logi.hist.plot(reviews$overall, reviews$helpfulYn, boxp = FALSE, type = hist,
               color = 'blue')
# Note: A nicely graphed empty model, haha



# Page 4 - Logistic Regression Setup in R ----

# Import and preview data
baseball = read.csv('/Users/hannah/Library/CloudStorage/GoogleDrive-gracesnouveaux@gmail.com/My Drive/Bethel Tech/Data Science/DSO106 Machine Learning and Modeling/1: Modeling and Optimization – Lesson 2. Modeling with Logistic Regression/baseball.csv')

View(baseball)
# 4,860 entries, 10 total columns


# Goal: Determine whether home runs predict a winning game score
  # IV (x axis, continuous): home runs
  # DV (y axis, categorical): winning game score
  # H0: Home runs do not predict a winning game score
    # H1: Home runs do predict a winning game score


# Wrangling

  # Recode DV
baseball$winsR = NA
baseball$winsR[baseball$W.L == 'W'] = 0
baseball$winsR[baseball$W.L == 'L'] = 1

View(baseball)
# 4,860 entries, 11 total columns


# Create logistic regression model and prep data for testing assumptions

  # Create model
baseLogM = glm(winsR ~ HR.Count, data = baseball, family = 'binomial')


  # Create probabilities variable then add to dataframe as converted to 
    # positive / negative predictions (aka: responses or probable outcomes)
baseProb = predict(baseLogM, type = 'response')
baseball$predicted = if_else(baseProb > 0.5, 'pos', 'neg')

View(baseball)
# 4,860 entries, 12 total columns


  # Recode predictions variable
baseball$predictedR = NA
baseball$predictedR[baseball$predicted == 'pos'] = 1
baseball$predictedR[baseball$predicted == 'neg'] = 0

View(baseball)
# 4,860 entries, 13 total columns


  # Convert predictions and DV to factors
baseball$predictedR = as.factor(baseball$predictedR)
baseball$winsR = as.factor(baseball$winsR)

str(baseball)
# Note: Success! Both are factors, with 2 levels each



# Test assumptions

  # Sample size
baseballCm = caret::confusionMatrix(baseball$predictedR, baseball$winsR)

baseballCm
# Confusion Matrix and Statistics
#           Reference
# Prediction    0    1
#          0 1190  513
#          1 1240 1917
#               Accuracy : 0.6393
# ...
  # Note: All cells have over 5 (and thus over 1) cases - validated!
    # IV successfully predicted DV ~64% of the time


  # Logit linearity

    # Create new dataset with only numeric variables
baseball2 = baseball %>% dplyr::select_if(is.numeric)

View(baseball2)
# 4,860 entries, 5 total columns


    # Create new variable with column names of the numeric data, to be treated
      # as multiple IV's (including home runs, as in original goal notes)
basePredictors = colnames(baseball2)

View(basePredictors)
# Note: Success

remove(basePredictors)
# Note: Just played around and tested removing this, because we don't end up
  # using it.. and that was fine - will ask on code review why we do this
  # step!


    # Create new dataframe to include a new column with the logit (calculated 
      # in this line of code using dataframe created above), as well as two 
      # columns for the predictors (IV’s) that were separate columns in the 
      # dataframe above – one for the column names and one for the values from 
      # those columns… this is converting the data from wide (one value per row, 
      # with each IV in a separate column) to long (one value per row with all 
      # IV’s in the same column 
baseball3 = baseball2 %>% 
  mutate(baseLogit = log(baseProb / (1 - baseProb))) %>% 
  gather(key = 'baseballPredictors', value = 'baseballPredictor.value', 
         -baseLogit)

View(baseball3)
# 24,300 entries, 3 total columns
  # Note: `, -baselogit` argument was intended to exclude the new column
    # after calculating the data, but that is not working for me... I also
    # don't understand why we would create it and immediately delete it in
    # the query...


    # Graph the logit
ggplot(baseball3, aes(baseLogit, baseballPredictor.value)) +
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = 'loess') +
  theme_bw() +
  facet_wrap(~baseballPredictors, scales = 'free_y')
# Note: HR count is the only one we need to validate, and it's strongly
  # linear :) 
  # As expected, this does not include the recoded DV - winsR - b/c we
  # converted it to be a factor before subsetting to only include numeric
  # data... but in the lesson, their graph includes it... so I'll reconvert 
  # to numeric and re-do some code above to include
  # Also... my results appear to be the opposite of the lesson for most
  # variables - Att is just totally different, and I'm not sure why in either
  # case...
  # Will be asking about both (missing recoded DV and different results) in
  # code review


      # Convert winsR back to numeric and redo other code to use it in this
        # graph
baseball2$winsR = as.numeric(baseball$winsR)

baseball3 = baseball2 %>% 
  mutate(baseLogit = log(baseProb / (1 - baseProb))) %>% 
  gather(key = 'baseballPredictors', value = 'baseballPredictor.value', 
         -baseLogit)

ggplot(baseball3, aes(baseLogit, baseballPredictor.value)) +
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = 'loess') +
  theme_bw() +
  facet_wrap(~baseballPredictors, scales = 'free_y')
# Note: The good news is I have the recoded DV now.. the bad news is it is 
  # opposite the results in the lesson (as with the others)


  # Multicollinearity - doesn't apply per only using the single Home Run IV


  # Independent errors

    # Graph model residuals
plot(baseLogM$residuals)
# Note: Per caring only about x axis, this assumption is validated
  # I have the same results as the lesson.. but reversed again... maybe
  # it's an R update since my code is the same? Will include this in my code
  # review question

    # Durbin-Watson test
dwtest(baseLogM, alternative = 'two.sided')
# 	Durbin-Watson test
# data:  baseLogM
# DW = 2.0828, p-value = 0.003875
# alternative hypothesis: true autocorrelation is not 0
  # Note: The p value indicates significance, which would violate the 
    # assumption, but the DW value is btw. 1-3 which means the assumption is, in
    # fact, validated - woo hoo!


  # Outliers - influential
baseballInfl = influence.measures(baseLogM)

summary(baseballInfl)
# Note: I don't see any issues with the results that printed, but there is a 
  # lot that didn't print... while I like this option, to actually be useful
  # it needs to be followed up with a query to limit the results to only print
  # those that are significant / actual outliers



# Page 5 - Logistic Regression in R ----

# Run analysis

  # Summary statistics for model
summary(baseLogM)
# Call:
# glm(formula = winsR ~ HR.Count, family = "binomial", data = baseball)
# Deviance Residuals: 
#     Min       1Q   Median       3Q      Max  
# -1.5338  -1.2389   0.3553   1.1171   2.5366  
# Coefficients:
#             Estimate Std. Error z value Pr(>|z|)    
# (Intercept)  0.80749    0.04658   17.34   <2e-16 ***
# HR.Count    -0.66398    0.03044  -21.81   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# (Dispersion parameter for binomial family taken to be 1)
#     Null deviance: 6737.4  on 4859  degrees of freedom
# Residual deviance: 6161.4  on 4858  degrees of freedom
# AIC: 6165.4
# Number of Fisher Scoring iterations: 4
  # Note: The p value for Home Runs is significant, which means it does predict
    # wins - reject the null and accept the alternative hypothesis; each home 
    # runs increases the odds for a winning score by 66% (that seems a bit
    # unlikely...)


  # Plot model
logi.hist.plot(baseball$HR.Count, baseball$winsR, boxp = FALSE, type = 'hist',
               col = 'blue')
# Note: Not sure how to interpret this.. the lesson's graph is different, and 
  # there are not notes to help (insert eyeroll) 