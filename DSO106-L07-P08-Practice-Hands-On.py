# %%
# DSO106 - Machine Learning and Modeling
    # Lesson 7 - k-Means and k-Nearest Neighbors%%
        # AKA: Machine Learning Lesson 2
    # Page 8 - Clustering Hands-On

# Requirements: Determine how cars are grouped together by using the mpg 
    # dataset built into Seaborn. Import it using the following code:
        # Mpg = sns.load_dataset('mpg')
    # If seaborn isn't working for you, click here to download the data.
    # Remember that you need continuous variables for these analyses, so 
        # you'll want to pinpoint columns such as mpg, cylinders, 
        # displacement, horsepower, weight, acceleration or model_year as 
        # variables.
    # Then use first the k-means machine learning algorithm to find the most 
        # appropriate k to examine, and provide the graph as well as add the 
        # cluster labels back into your dataframe. How are these groups being 
        # divided? What conclusions can you draw about the data?

# Goal: Determine how cars are grouped together

# Import potential packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# %%

# Import data
mpgOriginal = sns.load_dataset('mpg')

mpgOriginal

# %%
# 398 rows × 9 columns
    # Note: origin and name are categorical and will need to be dropped...
        # technically model year is too, and probably cylinders, but since 
        # it's numeric we can keep it in, I guess (requirements say to do so)


# Wrangling

    # Drop origin and name
mpg = mpgOriginal.drop(['origin', 'name'], axis = 1)

mpg

# %%
# 398 rows × 7 columns

# k-Means Analysis with 2 clusters

    # Create k-Means model
mpgKm = KMeans(n_clusters = 2)

# %%

    # Fit the data to the model
mpgKm.fit(mpg)

# %%
# ValueError: Input contains NaN, infinity or a value too large for dtype
        # ('float64').
    # Note: Apparently I need to remove missing values... and possibly can't
        # have float datatype...

        # Remove missing values
mpg.dropna(inplace = True)

mpg

# %%
# 392 rows × 7 columns
    # Note: Dropped 6 rows... will try to fit the model again, to confirm
        # whether floats are also an issue

mpgKm.fit(mpg)
        
# %%
# KMeans(n_clusters=2)
    # Note: Seems like just removing the missing values was enough to resolve
        # the error
    

# Interpretation with 2 clusters

    # Plot mpg and cylinders
plt.figure(figsize = (10, 6))
plt.title('k Means')
plt.xlabel('mpg')
plt.ylabel('cylinders')
plt.scatter(mpg.mpg, mpg.cylinders, c = mpgKm.labels_, cmap = 'plasma')

# %%
# Note: Some overlap between about 18-32 mpg and 4-6 cylinders

    # Plot mpg and displacement
plt.figure(figsize = (10, 6))
plt.title('k Means')
plt.xlabel('mpg')
plt.ylabel('displacement')
plt.scatter(mpg.mpg, mpg.displacement, c = mpgKm.labels_, cmap = 'plasma')

# %%
# Note: Some overlap between about 18-32 mpg again and 125-200 units    
    # displacement

    # Plot mpg and horsepower
plt.figure(figsize = (10, 6))
plt.title('k Means')
plt.xlabel('mpg')
plt.ylabel('horsepower')
plt.scatter(mpg.mpg, mpg.horsepower, c = mpgKm.labels_, cmap = 'plasma')

# %%
# Note: Some overlap between about 20-31 mpg and 62-115 horsepower

    # Plot mpg and weight
plt.figure(figsize = (10, 6))
plt.title('k Means')
plt.xlabel('mpg')
plt.ylabel('weight')
plt.scatter(mpg.mpg, mpg.weight, c = mpgKm.labels_, cmap = 'plasma')

# %%
# Note: Some overlap between about 18-32 mpg again, but there is no overlap in 
    # weight, the dividing line is at about 3k

    # Plot mpg and acceleration
plt.figure(figsize = (10, 6))
plt.title('k Means')
plt.xlabel('mpg')
plt.ylabel('acceleration')
plt.scatter(mpg.mpg, mpg.acceleration, c = mpgKm.labels_, cmap = 'plasma')

# %%
# Note: Some overlap between about 18-32 mpg again and 11-23 units acceleration

    # Plot mpg and model year
plt.figure(figsize = (10, 6))
plt.title('k Means')
plt.xlabel('mpg')
plt.ylabel('model year')
plt.scatter(mpg.mpg, mpg.model_year, c = mpgKm.labels_, cmap = 'plasma')

# %%
# Note: Some overlap between about 14-30 mpg and 70-81 model years


    # Add labels to data
mpg['Category'] = mpgKm.labels_

mpg

# %%
# 392 rows × 8 columns

        # Investigate means by category
mpg.groupby('Category').mean()

# %%
# Notes:
    # 0 = less mpg, more cylinders, higher displacement and horsepower, heavier
        # weight, slower acceleration, older
    # 1 = more mpg, less cylinders, lower displacement and horsepower, lighter
        # weight, faster acceleration, newer
    # These all make sense to be grouped together


# k-Means Analysis with 3 clusters

    # Create k-Means model
mpgKm2 = KMeans(n_clusters = 3)

# %%

    # Fit the data to the model
mpgKm2.fit(mpg)

# %%
# KMeans(n_clusters=3)


# Interpretation with 3 clusters

    # Add labels to data
mpg['Category2'] = mpgKm2.labels_

mpg

# %%
# 392 rows × 9 columns

        # Investigate means by category
mpg.groupby('Category2').mean()

# %%
# Notes:
    # 0 = lowest mpg (1/2 of category 2), highest cylinders, highest 
        # displacement (over 3x cat. 1), highest horsepower (over 2x cat. 1),
        # highest weight (almost 2x cat. 1), slowest accleration, oldest
    # 1 = highest mpg (2x cat. 2), lowest cylinders, lowest displacement (less 
        # than 1/3 cat. 0), lowest horsepower (less than 1/2 cat. 0), lowest 
        # weight (almost 1/2 cat. 0), fastest accleration, newest
    # 2 = mid-range mpg, mid-range cylinders, mid-range displacement, mid-range 
        # horsepower, mid-range weight, mid-range accleration, mid-range year - 
        # though actually very close to cat. 1
    # Adding a third group seems valuable, per finding enough differentiation
        # btw. them compared w/ only two groups


# k-Means Analysis with 4 clusters

    # Create k-Means model
mpgKm3 = KMeans(n_clusters = 4)

# %%

    # Fit the data to the model
mpgKm3.fit(mpg)

# %%
# KMeans(n_clusters=3)


# Interpretation with 3 clusters

    # Add labels to data
mpg['Category3'] = mpgKm3.labels_

mpg

# %%
# 392 rows × 10 columns

        # Investigate means by category
mpg.groupby('Category3').mean()

# %%
# Notes:
    # 0 = mid-range mpg, low cylinders, mid-range displacement, low horsepower, 
        # mid-range weight, mid-range accleration, newest
    # 1 = lowest mpg (about 1/2 of cat. 2), highest cylinders (2x cat. 2), 
        # highest displacement (about 3 1/2x cat. 2), highest horsepower (over 
        # 2x cat. 2), highest weight (over 2x cat. 2), slowest accleration, 
        # oldest
    # 2 = highest mpg (about 2x of cat. 1), lowest cylinders (1/2 cat. 1), 
        # lowest displacement (about 1/4 cat. 1), lowest horsepower (less than 
        # 1/2 cat. 1), lowest weight (less than 1/2 cat. 1), fastest 
        # accleration, mid-range year
    # 3 = mid-range mpg, mid-range cylinders, mid-range displacement, mid-range 
        # horsepower, mid-range weight, mid-range accleration, mid-range year
    # Adding a fourth group doesn't seem valuable, per not having enough 
        # differentiation between groups 0 and 3

# %%
# Summary: This model seems best at grouping data when it is clustering into
    # three groups: 1) older, heavier, slower cars; 2) mid-range cars; 3) 
    # newer, lighter, faster cars