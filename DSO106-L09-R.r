# DSO106 - Machine Learning and Modeling
  # Lesson 9 - Bayesian Networks
    # AKA: Machine Learning Lesson 4

# Page 5 - A/B Testing

# Goal: Determine whether your existing recipe for cream cheese frosting (A) is 
    # better than your new recipe for cream cheese frosting (B)
  # H0: Customers like them both the same
    # H1: Customers like one better than the other


# Data collected:
#  Frosting Type	Ate it All	Did Not Eat it All	Ratio
#       Old	          95	            22	        .82
#       New	          73	            46	        .61



# Define prior probability
  # Assume that 80% of all bakesale buyers will finish eating your cupcakes 
  # with both types of frosting


# Define number of trials
trials = 10000


# Define alpha (people ate the whole cupcake) and beta (people only ate part)
alpha = 8
beta = 2


# Create mock data using Monte Carlo Simulation
  # A = old frosting / B = new frosting
samplesA = rbeta(trials, 95 + alpha, 22 + beta)
samplesB = rbeta(trials, 73 + alpha, 46 + beta)


# Confirm how many times B was better than A
bBest = sum((samplesB > samplesA) / trials)

print(bBest)
# [1] 6e-04
  # Note: This validates that there were so few times people finished a cupcake
    # with the new frosting that the original frosting is best!