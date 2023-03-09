# %%
# DSO106 - Machine Learning and Modeling
    # Lesson 10 - Natural Language Processing
        # AKA: Machine Learning Lesson 5

# Page 1 - Introduction

    # From Workshop

# Import packages
import matplotlib.pyplot as plt
import nltk
import requests
import seaborn as sns
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer

# %%

# Read in data

    # Set up URL variable to be able to import from a URL
prideNPredjUrl = 'https://www.gutenberg.org/files/1342/1342-0.txt'

# %%

    # Set up function to get data from the URL
pridNPredjOriginalText = requests.get(prideNPredjUrl)

type(pridNPredjOriginalText)

# %%
# requests.models.Response


# Set up variables to parse data
prideNPredjHtml = pridNPredjOriginalText.text
prideNPredjSoup = BeautifulSoup(prideNPredjHtml, 'html.parser')

type(prideNPredjSoup)

# %%
# bs4.BeautifulSoup


# Create variable to tokenize the text by word
prideNPredjText = soup.get_text()

prideNPredjTokenizer = RegexpTokenizer('\w+')

prideNPredjTokens = prideNPredjTokenizer.tokenize(prideNPredjText)

# %%

    # View first five words
prideNPredjTokens[:5]

# %%
# ['ï', 'The', 'Project', 'Gutenberg', 'eBook']

    # View first 25 words
prideNPredjTokens[:25]

# %%
# ['ï',
#  'The',
#  'Project',
#  'Gutenberg',
#  'eBook',
#  'of',
#  'Pride',
#  'and',
#  'prejudice',
#  'by',
#  'Jane',
#  'Austen',
#  'This',
#  'eBook',
#  'is',
#  'for',
#  'the',
#  'use',
#  'of',
#  'anyone',
#  'anywhere',
#  'in',
#  'the',
#  'United',
#  'States']
    # Note: Words are recognized as the same by case, so it's important to have
        # everything in the same case


# Update all words to be lowercase
prideNPredjWords = []

for word in tokens:
    prideNPredjWords.append(word.lower())

prideNPredjWords[:5]

# %%
# ['ï', 'the', 'project', 'gutenberg', 'ebook']
    # Note: Success!


# Remove stop words

    # Create list of stop words
nltk.download('stopwords')

prideNPredjStopwords = nltk.corpus.stopwords.words('english')

prideNPredjStopwords[:5]

# %%
# ['i', 'me', 'my', 'myself', 'we']

    # Create variable with text, removing stop words
prideNPredjWordsWOStops = []

for word in prideNPredjWords:
    if word not in prideNPredjStopwords:
        prideNPredjWordsWOStops.append(word)

prideNPredjWordsWOStops[:10]

# %%
# ['ï',
#  'project',
#  'gutenberg',
#  'ebook',
#  'pride',
#  'prejudice',
#  'jane',
#  'austen',
#  'ebook',
#  'use']


# Plot top 25 words
sns.set()
prideNPredjWordFreq = nltk.FreqDist(prideNPredjWordsWOStops)
prideNPredjWordFreq.plot(25)

# %%
# Note: Data needs more wrangling, but definitely fun to see elizabeth, darcy, 
    # bennet, bingley, etc.

# %%

# Page 3 - Read in Text

    # Create a variable to hold the actual URL
ctMnteCristoUrl = 'http://www.gutenberg.org/files/1184/1184-h/1184-h.htm'

    # Read in the text from a website
ctMnteCristoOriginalText = requests.get(ctMnteCristoUrl)

# %%

# Page 4 - Convert Text to Soup

# Extract ALL text from URL (ie: including html tags, etc.)
ctMnteCristoHtml = ctMnteCristoOriginalText.text

# %%

# Parse the extracted text - make soup!
ctMnteCristoSoup = BeautifulSoup(ctMnteCristoHtml, 'html.parser')

# %%

# Use HTML tags to extract useful info

    # Call the `title` tag, with results returned as a `string`
ctMnteCristoSoup.title.string

# %%
# 'The Project Gutenberg eBook of The Count of Monte Cristo, by Alexandre 
    # Dumas, pÃ¨re'

# %%

# Page 5 - Tokenize Data

# Retrieve text
ctMnteCristoText = ctMnteCristoSoup.get_text()

# %%

    # Create tokenizer object to separate text into words
ctMnteCristoTokenizer = RegexpTokenizer('\w+')

# %%

    # Split text into tokens / words
ctMnteCristoTokens = ctMnteCristoTokenizer.tokenize(ctMnteCristoText)

ctMnteCristoTokens[:5]

# %%
# ['The', 'Project', 'Gutenberg', 'eBook', 'of']

# %%

# Page 6 - Remove Capitalization

# Create variable to hold lowercase words
ctMnteCristoWords = []

# %%

    # Convert to lowercase with new function
for word in ctMnteCristoTokens:
    ctMnteCristoWords.append(word.lower())

ctMnteCristoWords[:5]

# %%
# ['the', 'project', 'gutenberg', 'ebook', 'of']
    # Note: Success!

# %%

# Page 7 - Remove Stopwords

# Create variable with only stopwords
ctMnteCristoStopwords = nltk.corpus.stopwords.words('english')

ctMnteCristoStopwords[:10]

# %%
# ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]


    # Create a variable for text with stopwords removed
ctMnteCristoWOstops = []

# %%

    # Create function to remove stopwords
for word in ctMnteCristoWords:
    if word not in ctMnteCristoStopwords:
        ctMnteCristoWOstops.append(word)

ctMnteCristoWOstops[:5]

# %%
# ['project', 'gutenberg', 'ebook', 'count', 'monte']

# %%

# Page 8 - Count and Plot Words

# Create a plot for the frequency of the top 25 words
sns.set()
ctMnteCristoWordFreq = nltk.FreqDist(ctMnteCristoWOstops)
ctMnteCristoWordFreq.plot(25)

# %%
# Note: Makes sense to see words like count, man, villefort, etc.!