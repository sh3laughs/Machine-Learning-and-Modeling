# %%
# DSO106 - Machine Learning and Modeling
    # Lesson 5 - Randomly Generating Data

# Page 3 - Generating a Uniform Distribution

# Import packages
import random
import numpy as np

# %%

    # Generating Numbers Between 0 and 1
random1 = random.random()

print(random1)

# %%
# 0.5148601064773253

    # Generating User-Specified Values
random2 = random.uniform(30, 50)

print(random2)

# %%
# 37.09801164790876

# %%

# Page 4 - Generating Random Numbers - Normal Distribution

    # Generating the Standard Normal Distribution
random3 = np.random.normal(size = 5)

print(random3)

# %%
# [ 0.88402425 -0.20599084  1.77683386 -0.38308628  0.37325743]

    # Generating a Normal Distribution Centered Elsewhere
random4 = (np.random.normal(size = 5)) * 10 + 50

print(random4)

# %%
# [46.2287225  48.22283749 47.57067085 61.09771053 50.99623481]