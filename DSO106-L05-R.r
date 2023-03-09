# DSO106 - Machine Learning and Modeling
  # Lesson 5 - Randomly Generating Data

# Page 3 - Generating a Uniform Distribution ----

  # Generating Numbers Between 0 and 1
runif(15)
# [1] 0.3849944 0.3914903 0.7242495 0.5442608 0.6633067 0.4734115 0.6028212 
  # 0.4029008
#  [9] 0.6562862 0.2586763 0.9792932 0.2989809 0.8547467 0.9667656 0.1762035


  # Generating User-Specified Values
runif(8, 12, 20)
# [1] 16.09220 14.91345 13.62718 13.45376 15.26817 14.02124 13.53282 18.95352



# Page 4 - Generating Random Numbers - Normal Distribution ----
random1 = rnorm(20, 35, 7)

print(random1)
#  [1] 20.43881 33.35203 29.11222 27.79425 33.40907 39.82361 28.89415 36.73275
#  [9] 42.85417 33.83455 48.06129 30.89703 37.22861 27.50152 44.96528 35.95130
# [17] 38.37016 33.38653 35.53181 26.85350