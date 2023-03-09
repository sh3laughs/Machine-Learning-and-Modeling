# %%
# DSO106 - Machine Learning and Modeling
    # Lesson 10 - Natural Language Processing
        # AKA: Machine Learning Lesson 5
    # Page 10 - Lesson 5 Hands-On

# Requirements: For this hands-on, you will be using Alice's Adventures in 
        # Wonderland by Lewis Carroll to practice your newfound NLP skills. The 
        # book can be found here. Follow the process you used on The Count of 
        # Monte Cristo to create a graphic of the most frequently used words in 
        # Alice's Adventures in Wonderland.
    # Please attach a Jupyter Notebook with your code, your graphic, and your   
        # conclusions.

# Import packages
import matplotlib.pyplot as plt
import nltk
import requests
import seaborn as sns
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer

# %%

# Read in Text

    # Create a variable to hold the actual URL
aiwUrl = 'https://www.gutenberg.org/files/11/11-h/11-h.htm'

# %%
    # Read in the text from a website
aiwOriginalText = requests.get(aiwUrl)

type(aiwOriginalText)

# %%
# requests.models.Response

# %%

# Convert Text to Soup

    # Extract ALL text from URL (ie: including html tags, etc.)
aiwHtml = aiwOriginalText.text

# %%

    # Parse the extracted text - make soup!
aiwSoup = BeautifulSoup(aiwHtml, 'html.parser')

type(aiwSoup)

# %%
# bs4.BeautifulSoup

# %%

# Tokenize Data

    # Retrieve text
aiwText = aiwSoup.get_text()

# %%

    # Create tokenizer object to separate text into words
aiwTokenizer = RegexpTokenizer('\w+')

# %%

    # Split text into tokens / words
aiwTokens = aiwTokenizer.tokenize(aiwText)

aiwTokens[:5]

# %%
# ['The', 'Project', 'Gutenberg', 'eBook', 'of']
    # Note: Success!

# %%

# Remove Capitalization

    # Create variable to hold lowercase words
aiwWords = []

# %%

    # Convert to lowercase with new function
for word in aiwTokens:
    aiwWords.append(word.lower())

aiwWords[:5]

# %%
# ['the', 'project', 'gutenberg', 'ebook', 'of']
    # Note: Success!

# %%

# Remove Stopwords

    # Create variable with only stopwords
stopwords = nltk.corpus.stopwords.words('english')

stopwords[:10]

# %%
# ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]


    # Create a variable for text with stopwords removed
aiwWOstops = []

# %%

    # Create function to remove stopwords
for word in aiwWords:
    if word not in stopwords:
        aiwWOstops.append(word)

aiwWOstops[:5]

# %%
# ['project', 'gutenberg', 'ebook', 'alice', 'adventures']
    # Note: Success!

# %%

# Create a plot for the frequency of the top 25 words
sns.set()
aiwWordFreq = nltk.FreqDist(aiwWOstops)
aiwWordFreq.plot(25)

# %%
# Note: This definitely needs more pre-processing - gutenberg and project show
        # up in the top 25 words, which is suspicious!, as well as non-English 
        # characters showing up in the list
    # That said it's good to see words like `alice`, `queen`, and `hatter` on 
        # this plot