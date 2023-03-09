# %%
# DSO106 - Machine Learning and Modeling
    # Lesson 9 - Bayesian Networks
        # AKA: Machine Learning Lesson 4

# Page 4 - Parts of Bayes Theorem

# Calculate the probability that your instructor's a dork, given that she 
        # cuddles her statistics book at night
    # A: your instructor's a dork
    # B: cuddling statistics books

    # Define likelihood: the probability that cuddling a statistics book is 
            # good evidence that your instructor's a dork
        # P(B|A) = 8/10
PBA = 8/10

PBA

# %%
# 0.8

    # Define prior: the probability that your instructor's a dork
        # P(A) = 6/10
PA = 6/10

PA

# %%    
# 0.6

    # Define normalizing factor: the probability of anyone cuddling statistics 
            # books
        # P(B) = .000065
PB = .000065

PB

# %%
# 6.5e-05

    # Calculate the Posterior
PAB = (PBA * PA) / PB

print(PAB)

# %%
# 7384.615384615385
    # Note: No way to interpret this without at least one more equation to 
        # compare it to


    # Define a new A to create a new model, for comparison, so that the results
            # above can be tested for value
        # A: your instructor doesn't own a pillow
    

    # Define prior: the probability that your instructor's a dork
        # P(A) = 1/100
PA2 = 1/100

PA2

# %%
# 0.01

    # Recalculate using other variables unchanged
PAB2 = (PBA * PA2) / PB

print(PAB2)

# %%
# 123.0769230769231
    # Note: This is so much lower than the original results that it validates
        # that the instructor cuddles her statistics book because she is a dork
        # and not because she is pillow-less