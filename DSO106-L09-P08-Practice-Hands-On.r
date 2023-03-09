# DSO106 - Machine Learning and Modeling
  # Lesson 9 - Bayesian Networks
    # AKA: Machine Learning Lesson 4
  # Page 8 - Lesson 4 Practice Hands-On

# Requirements: For this hands-on, you will be determining which type of mold-
    # removal solution works better: just bleaching objects, or bleaching them 
    # and scrubbing them down thoroughly out of 10,000 trials. Based on the 
    # priors you created, the mold-removal solutions have a 90% chance of 
    # working.
  # You're trying to determine whether the mold will grow back or not, using 
    # the following table:
      # Mold Removal Type	 Mold Returned	 Did Not Return	 Ratio
      #     Bleach	            27	            39	        .41
      # Bleach and Scrubbing	  10	            45	        .18
  # Complete A/B testing and Monte Carlo simulation using R. Please attach your 
    # R script file with your code documented and information in comments about 
    # your findings.


# Define number of trials
trials = 10000


# Define alpha (mold returned) and beta (mold didn't return)
alpha = 9
beta = 1


# Create mock data using Monte Carlo Simulation
  # A = bleach only / B = bleach and scrubbing
samplesA = rbeta(trials, 27 + alpha, 39 + beta)
samplesB = rbeta(trials, 10 + alpha, 45 + beta)


# Confirm how many times B was better than A
bBest = sum((samplesB > samplesA) / trials)

print(bBest)
# [1] 0.0115
  # Note: I think this is saying that bleach and scrubbing is less effective
    # than just bleach... which doesn't make sense with the data (or IRL) will
    # ask on a code review
    # Also... my results are different than the solution, and I'm not sure if 
    # that is b/c the randomized data will always yield differing results