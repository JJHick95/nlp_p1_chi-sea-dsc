
# Agenda today:
1. Overview of NLP
2. Model Building Remains Consistent
2. Pre-Processing for NLP 
    - Tokenization
    - Stopwords removal
    - Lexicon normalization: lemmatization and stemming
3. Feature Engineering for NLP
    - Bag-of-Words
    - Count Vectorizer
    - Term frequency-Inverse Document Frequency (tf-idf)



```python
# This is always a good idea
%load_ext autoreload
%autoreload 2

# This is always a good idea
%load_ext autoreload
%autoreload 2

from src.student_caller import one_random_student
from src.student_list import student_first_names
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


## 1. Overview of NLP
NLP allows computers to interact with text data in a structured and sensible way. In short, we will be breaking up series of texts into individual words (or groups of words), and isolating the words with **semantic value**.  We will then compare texts with similar distributions of these words, and group them together.

In this section, we will discuss some steps and approaches to common text data analytic procedures. In other words, with NLP, computers are taught to understand human language, its meaning and sentiments. Some of the applications of natural language processing are:
- Chatbots 
- Speech recognition and audio processing 
- Classifying documents 

Here is an example that uses some of the tools we use in this notebook.  
  -[chi_justice_project](https://chicagojustice.org/research/justice-media-project/)  
  -[chicago_justice classifier](https://github.com/chicago-justice-project/article-tagging/blob/master/lib/notebooks/bag-of-words-count-stemmed-binary.ipynb)

We will introduce you to the preprocessing steps, feature engineering, and other steps you need to take in order to format text data for machine learning tasks. 

We will also introduce you to [**NLTK**](https://www.nltk.org/) (Natural Language Toolkit), which will be our main tool for engaging with textual data.

# NLP process 
<img src="img/nlp_process.png" style="width:1000px;">

# 2. Model Building Principles Remain Consistent


```python
#!pip install nltk
# conda install -c anaconda nltk
```

We will be working with a dataset which includes both **satirical** (The Onion) and real news (Reuters) articles. 

We refer to the entire set of articles as the **corpus**.  


```python
# Let's import our corpus. 
import pandas as pd

```

Our goal is to detect satire, so our target class of 1 is associated with The Onion articles.  

![the_onion](img/the_onion.jpeg) ![reuters](img/reuters.png)

Each article in the corpus is refered to as a **document**.


```python
# How many documents are there in the corpus?
```


```python
# What is the class balance?
```


```python
# Let's look at some example texts from both categories

# Category 1

```


```python
# Category 0
```

Let's think about our types of error and the use cases of being able to correctly separate satirical from authentic news. What type of error should we decide to optimize our models for?  


```python
# Help me fill in the blanks:

# Pass in the body of the documents as raw text
X = None
y = None
```


```python
X_train, X_test, y_train, y_test = None, None, None, None
```


```python
# For demonstration purposes, we will perform a secondary train test split. In practice, we will apply a pipeline.

X_t, X_val, y_t, y_val = train_test_split(X_train, y_train, random_state=42, test_size=.2)

```


```python
from sklearn.feature_extraction.text import TfidfVectorizer

# A new preprocessing tool!
tfidf = TfidfVectorizer(token_pattern=r"([a-zA-Z]+(?:'[a-z]+)?)")

# Like always, we are fitting only on the training set
X_t_vec = None
# Here is our new dataframe of predictors
X_t_vec = pd.DataFrame(X_t_vec.toarray(), columns = tfidf.get_feature_names())

X_train_vec.head()
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-196-011e465d70fb> in <module>
          7 X_t_vec = None
          8 # Here is our new dataframe of predictors
    ----> 9 X_t_vec = pd.DataFrame(X_t_vec.toarray(), columns = tfidf.get_feature_names())
         10 
         11 X_train_vec.head()


    AttributeError: 'NoneType' object has no attribute 'toarray'



```python
# That's a lot of columns
X_t_vec.shape
```




    (640, 19118)



### We can push this data into any of our classification models



```python
# Code here
```

# We proceed as usual:
    - Transform the validation set
    - Score the validation set
    - Plot a confusion matrix to check out the performance on different types of errors
 


```python
# your code here

# Transform X_val
X_val_vec = None

# Score val
y_hat = None

# Plot confusion matrix

```

# 3 Preprocessing

### Tokenization 

In order to convert the texts into data suitable for machine learning, we need to break down the documents into smaller parts. 

The first step in doing that is **tokenization**.

Tokenization is the process of splitting documents into units of observations. We usually represent the tokens as __n-gram__, where n represent the consecutive words occuring in a document. In the case of unigram (one word token), the sentence "David works here" can be tokenized into?

"David", "works", "here"
"David works", "works here"

Let's consider the first document in our corpus:


```python
first_document = corpus.iloc[0].body
```

There are many ways to tokenize our document. 

It is a long string, so the first way we might consider is to split it by spaces.


```python
# code
```


```python
print(f'{one_random_student(mccalister)}, in creating tokens in this way, what problems do we see?')
```

#### Chat out some problems (don't look down)

<img src="https://media.giphy.com/media/ZaiC2DYDRiqhQ269nz/giphy.gif" style="width:1500px;">

We are trying to create a set of tokens with **high semantic value**.  In other words, we want to isolate text which best represents the meaning in each document.  


## Common text cleaning tasks:  
  1. remove capitalization  
  2. remove punctuation  
  3. remove stopwords  
  4. remove numbers

We could manually perform all of these tasks with string operations

## Capitalization

When we create our matrix of words associated with our corpus, **capital letters** will mess things up.  The semantic value of a word used at the beginning of a sentence is the same as that same word in the middle of the sentence.  In the two sentences:

sentence_one =  "Excessive gerrymandering in small counties suppresses turnout."   
sentence_two =  "Turnout is suppressed in small counties by excessive gerrymandering."  

Excessive has the same semantic value, but will be treated as two separate tokens because of capitals.


```python
sentence_one =  "Excessive gerrymandering in small counties suppresses turnout." 
sentence_two =  "Turnout is suppressed in small counties by excessive gerrymandering."

```


```python
sentence_one =  "Excessive gerrymandering in small counties suppresses turnout." 
sentence_two =  "Turnout is suppressed in small counties by excessive gerrymandering."
```

### Let's fill in the list comprehension below to manually and remove capitals from the 1st document


```python
manual_cleanup = None
```


```python
print(f"Our initial token set for our first document is {len(manual_cleanup)} words long")
```

    Our initial token set for our first document is 154 words long



```python
print(f"Our initial token set for our first document has {len(set(first_document.split()))} unique words")
```

    Our initial token set for our first document has 117 unique words



```python
print(f"After remove caps, our first document has {len(set(manual_cleanup))} unique words")
```

    After remove caps, our first document has 115 unique words


## Punctuation

Like capitals, splitting on white space will create tokens which include punctuation that will muck up our semantics.  

Returning to the above example, 'gerrymandering' and 'gerrymandering.' will be treated as different tokens.

# Different ways to strip punctuation


```python
# Strip with translate

## Manual removal of punctuation
# string library!
import string

string.punctuation
punctuation = string.punctuation + '“'
punctuation
# string.ascii_letters


```




    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~“'



# Strip with regex

To remove them, we will use regular expressions, a powerful tool which you may already have some familiarity with.

Regex allows us to match strings based on a pattern.  This pattern comes from a language of identifiers, which we can begin exploring on the cheatsheet found here:
  -   https://regexr.com/

A few key symbols:
  - . : matches any character
  - \d, \w, \s : represent digit, word, whitespace  
  - *, ?, +: matches 0 or more, 0 or 1, 1 or more of the preceding character  
  - [A-Z]: matches any capital letter  
  - [a-z]: matches lowercase letter  


```python
manual_cleanup
```




    ['noting',
     'that',
     'the',
     'resignation',
     'of',
     'james',
     'mattis',
     'as',
     'secretary',
     'of',
     'defense',
     'marked',
     'the',
     'ouster',
     'of',
     'the',
     'third',
     'top',
     'administration',
     'official',
     'in',
     'less',
     'than',
     'three',
     'weeks',
     'a',
     'worried',
     'populace',
     'told',
     'reporters',
     'friday',
     'that',
     'it',
     'was',
     'unsure',
     'how',
     'many',
     'former',
     'trump',
     'staffers',
     'it',
     'could',
     'safely',
     'reabsorb',
     '“jesus',
     'we',
     'can’t',
     'just',
     'take',
     'back',
     'these',
     'assholes',
     'all',
     'at',
     'once—we',
     'need',
     'time',
     'to',
     'process',
     'one',
     'before',
     'we',
     'get',
     'the',
     'next”',
     'said',
     '53yearold',
     'gregory',
     'birch',
     'of',
     'naperville',
     'il',
     'echoing',
     'the',
     'concerns',
     'of',
     '323',
     'million',
     'americans',
     'in',
     'also',
     'noting',
     'that',
     'the',
     'country',
     'was',
     'only',
     'now',
     'truly',
     'beginning',
     'to',
     'reintegrate',
     'former',
     'national',
     'security',
     'advisor',
     'michael',
     'flynn',
     '“this',
     'is',
     'just',
     'not',
     'sustainable',
     'i’d',
     'say',
     'we',
     'can',
     'handle',
     'maybe',
     'one',
     'or',
     'two',
     'more',
     'former',
     'members',
     'of',
     'trump’s',
     'inner',
     'circle',
     'over',
     'the',
     'remainder',
     'of',
     'the',
     'year',
     'but',
     'that’s',
     'it',
     'this',
     'country',
     'has',
     'its',
     'limits”',
     'the',
     'us',
     'populace',
     'confirmed',
     'that',
     'they',
     'could',
     'not',
     'handle',
     'all',
     'of',
     'these',
     'pieces',
     'of',
     'shit',
     'trying',
     'to',
     'rejoin',
     'society',
     'at',
     'once']




```python
# Test out a word and search it


```


```python
# Create a pattern that matches only letters to strip punctuation

# Code here
```


```python
print(f"After removing punctuation, our first document has {len(set(manual_cleanup))} unique words")
```

    After removing punctuation, our first document has 75 unique words


### Stopwords

Stopwords are the **filler** words in a language: prepositions, articles, conjunctions. They have low semantic value, and almost always need to be removed.  

Luckily, NLTK has lists of stopwords ready for our use.


```python
from nltk.corpus import stopwords
stopwords.words('english')[:10]
```




    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]




```python
stopwords.words('spanish')[:10]
```




    ['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se']



Let's see which stopwords are present in our first document.


```python
stops = [token for token in manual_cleanup if token in stopwords.words('english')]

```


```python
stops = [token for token in manual_cleanup if token in stopwords.words('english')]
stops[:10]
```




    ['that', 'the', 'of', 'as', 'of', 'the', 'of', 'the', 'in', 'than']




```python
print(f'There are {len(stops)} stopwords in the first document')
```

    There are 68 stopwords in the first document



```python
print(f'That is {len(stops)/len(manual_cleanup): .2%} of our text')
```

    That is  44.44% of our text


Let's also use the **FreqDist** tool to look at the makeup of our text before and after removal


```python
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
```


```python
fdist = FreqDist(manual_cleanup)
plt.figure(figsize=(10,10))
fdist.plot(30)

```


![png](index_files/index_67_0.png)





    <matplotlib.axes._subplots.AxesSubplot at 0x1a70f33198>




```python
manual_cleanup = [token for token in manual_cleanup if token not in stopwords.words('english')]
```


```python
# We can also customize our stopwords list

custom_sw = stopwords.words('english')
custom_sw.extend(["i'd","say"] )
custom_sw[-10:]
```




    ['wasn',
     "wasn't",
     'weren',
     "weren't",
     'won',
     "won't",
     'wouldn',
     "wouldn't",
     "i'd",
     'say']




```python
manual_cleanup = [token for token in manual_cleanup if token not in custom_sw]

```


```python
print(f'After removing stopwords, there are {len(set(manual_cleanup))} unique words left')
```

    After removing stopwords, there are 75 unique words left



```python
fdist = FreqDist(manual_cleanup)
plt.figure(figsize=(10,10))
fdist.plot(30)
```


![png](index_files/index_72_0.png)





    <matplotlib.axes._subplots.AxesSubplot at 0x1a70f30860>




```python
print(f'''{one_random_student(mccalister)}, based on the frequency plot above, 
    add some custome stopwords to our list, and remove them from our document. 
    Print out the frequency plot to confirm they have been removed.''')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-378-fa0ea6476d87> in <module>
          1 print(f'''{one_random_student(mccalister)}, based on the frequency plot above, 
          2     add some custome stopwords to our list, and remove them from our document.
    ----> 3     Print out the frequency plot to confirm they have been removed.''')
    

    NameError: name 'mccalister' is not defined



```python
# Code
```

#### Numbers

Numbers also usually have low semantic value. Their removal can help improve our models. 

To remove them, we will use regular expressions, a powerful tool which you may already have some familiarity with.

Regex allows us to match strings based on a pattern.  This pattern comes from a language of identifiers, which we can begin exploring on the cheatsheet found here:
  -   https://regexr.com/

A few key symbols:
  - . : matches any character
  - \d, \w, \s : represent digit, word, whitespace  
  - *, ?, +: matches 0 or more, 0 or 1, 1 or more of the preceding character  
  - [A-Z]: matches any capital letter  
  - [a-z]: matches lowercase letter  

Other helpful resources:
  - https://regexcrossword.com/
  - https://www.regular-expressions.info/tutorial.html

We can use regex to isolate numbers



# Individual Code Exercise:  
Individually, test out different regex patterns to isolate numerical values in the first document.



```python
first_document
```


```python
import re
pattern = 'a regex pattern'
number = re.findall( pattern, first_document)
number
```


```python
number
```

Sklearn and NLTK provide us with a suite of **tokenizers** for our text preprocessing convenience.


```python
import nltk
from nltk.tokenize import regexp_tokenize, word_tokenize, RegexpTokenizer

```


```python
first_document
```


```python
import re
re.findall(r"([a-zA-Z]+(?:'[a-z]+)?)" , "I'd")
```


```python
pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
tokenizer = RegexpTokenizer(r"([a-zA-Z]+(?:[’'][a-z]+)?)")
first_doc = tokenizer.tokenize(first_document)
first_doc = [token.lower() for token in first_doc]
first_doc = [token for token in first_doc if token not in custom_sw]
first_doc[10]
```


```python
print(f'We are down to {len(set(first_doc))} unique words')
```

# Stemming

Most of the semantic meaning of a word is held in the root, which is usually the beginning of a word.  Conjugations and plurality do not change the semantic meaning. "eat", "eats", and "eating" all have essentially the same meaning packed into eat.   

Stemmers consolidate similar words by chopping off the ends of the words.

![stemmer](img/stemmer.png)

There are different stemmers available.  The two we will use here are the **Porter** and **Snowball** stemmers.  A main difference between the two is how agressively it stems, Porter being less agressive.


```python
from nltk.stem import *

p_stemmer = PorterStemmer()
s_stemmer = SnowballStemmer(language="english")
```


```python
p_stemmer.stem(first_doc[0])
```


```python
s_stemmer.stem(first_doc[0])
```


```python
for word in first_doc:
    p_word = p_stemmer.stem(word)
    s_word = s_stemmer.stem(word)
    
    if p_word != s_word:
        print(p_word, s_word)
    
```


```python
first_doc = [p_stemmer.stem(word) for word in first_doc]
```


```python
fdist = FreqDist(first_doc)
plt.figure(figsize=(10,10))
fdist.plot(30)
```


```python
print(f'Stemming did not reduce our token count: {len(set(first_doc))} unique tokens')
```

# Lemming

Lemming is a bit more sophisticated that the stem choppers.  Lemming uses part of speech tagging to determine how to transform a word.  In that 
Lemmatization returns real words. For example, instead of returning "movi" like Porter stemmer would, "movie" will be returned by the lemmatizer.

- Unlike Stemming, Lemmatization reduces the inflected words properly ensuring that the root word belongs to the language.  It can handle words such as "mouse", whose plural "mice" the stemmers would not lump together with the original. 

- In Lemmatization, the root word is called Lemma. 

- A lemma (plural lemmas or lemmata) is the canonical form, dictionary form, or citation form of a set of words.

![lemmer](img/lemmer.png)



```python
from nltk.stem import WordNetLemmatizer 
  
lemmatizer = WordNetLemmatizer() 

```


```python
print(f'Mice becomes: {lemmatizer.lemmatize("mice")}')
print(f'Noting becomes: {lemmatizer.lemmatize(first_doc[0])}')
```


```python
# However, look at the output below:
    
sentence = "He saw the trees get sawed down"
lemmed_sentence = [lemmatizer.lemmatize(token) for token in sentence.split(' ')]
lemmed_sentence
```


```python
one_random_student(mccalister)
# What should have changed form but didn't?
```

Lemmatizers depend on POS tagging, and defaults to noun.


```python
one_random_student(mccalister)
# What do you think will happen if we feed another POS to saw?
```

With a little bit of work, we can POS tag our text.


```python
pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
tokenizer = RegexpTokenizer(r"([a-zA-Z]+(?:[’'][a-z]+)?)")
first_doc = tokenizer.tokenize(first_document)
first_doc = [token.lower() for token in first_doc]
first_doc = [token for token in first_doc if token not in custom_sw]
corpus.loc[0].body
```


```python
nltk.download('tagsets')
nltk.help.upenn_tagset()
```


```python
from nltk import pos_tag
# Use nltk's pos_tag to tag our words
# Does a pretty good job, but does make some mistakes
first_doc_tagged = pos_tag(first_doc)
first_doc_tagged
```


```python
# Then transform the tags into the tags of our lemmatizers
from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):
    '''
    Translate nltk POS to wordnet tags
    '''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
```


```python
first_doc_tagged = [(token[0], get_wordnet_pos(token[1]))
             for token in first_doc_tagged]
```


```python
first_doc_lemmed = [lemmatizer.lemmatize(token[0], token[1]) for token in first_doc_tagged]
```


```python
first_doc_lemmed[:10]
```


```python
print(f'There are {len(set(first_doc_lemmed))} unique lemmas')
```


```python
fdist = FreqDist(first_doc_lemmed)
plt.figure(figsize=(10,10))
fdist.plot(30)
```

## Part 3. Feature Engineering for NLP 
The machine learning algorithms we have encountered so far represent features as the variables that take on different value for each observation. For example, we represent individual with distinct education level, income, and such. However, in NLP, features are represented in very different way. In order to pass text data to machine learning algorithm and perform classification, we need to represent the features in a sensible way. One such method is called **Bag-of-words (BoW)**. 

A bag-of-words model, or BoW for short, is a way of extracting features from text for use in modeling. A bag-of-words is a representation of text that describes the occurrence of words within a document. It involves two things:

- A vocabulary of known words.
- A measure of the presence of known words.

It is called a “bag” of words, because any information about the order or structure of words in the document is discarded. The model is only concerned with whether known words occur in the document, not where in the document. The intuition behind BoW is that a document is similar to another if they have similar contents. Bag of Words method can be represented as **Document Term Matrix**, or Term Document Matrix, in which each column is an unique vocabulary, each observation is a document. For example:

- Document 1: "I love dogs"
- Document 2: "I love cats"
- Document 3: "I love all animals"
- Document 4: "I hate dogs"


Can be represented as:

![document term matrix](img/document_term_matrix.png)


```python
# implementing it in python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
# Convert a collection of text documents to a matrix of token counts

vec = CountVectorizer()
X = vec.fit_transform([" ".join(first_doc_lemmed)])


df = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
df.head()
```

That is not very exciting for one document. The idea is to make a document term matrix for all of the words in our corpus.


```python
corpus
```


```python
vec = CountVectorizer(token_pattern=r"([a-zA-Z]+(?:'[a-z]+)?)", stop_words=custom_sw)
X = vec.fit_transform(corpus.body[0:2])

df = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
df.head()
```


```python
vec = CountVectorizer(token_pattern=r"([a-zA-Z]+(?:'[a-z]+)?)", stop_words=custom_sw, ngram_range=[1,2])
X = vec.fit_transform(corpus.body[0:2])

df = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
df.head()
```

Our document term matrix gets bigger and bigger, with more and more zeros, becoming sparser and sparser.


```python
vec = CountVectorizer(token_pattern=r"([a-zA-Z]+(?:'[a-z]+)?)", stop_words=custom_sw, ngram_range=[1,2])
X = vec.fit_transform(corpus.body)

df = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
df.head()
```

We can set upper and lower limits to the word frequency.


```python
corpus.body
```


```python
vec = CountVectorizer(token_pattern=r"([a-zA-Z]+(?:'[a-z]+)?)", stop_words=custom_sw, ngram_range=[1,2], min_df=2, max_df=25)
X = vec.fit_transform(corpus.body)

df_cv = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
df_cv
```

### TF-IDF 
There are many schemas for determining the values of each entry in a document term matrix, and one of the most common schema is called the TF-IDF -- term frequency-inverse document frequency. Essentially, tf-idf *normalizes* the raw count of the document term matrix. And it represents how important a word is in the given document. 

> The goal of using tf-idf instead of the raw frequencies of occurrence of a token in a given document is to scale down the impact of tokens that occur very frequently in a given corpus and that are hence empirically less informative than features that occur in a small fraction of the training corpus.

- TF (Term Frequency)
term frequency is the frequency of the word in the document divided by the total words in the document.

- IDF (inverse document frequency)
IDF represents the measure of how much information the word provides, i.e., if it's common or rare across all documents. It is the logarithmically scaled inverse fraction of the documents that contain the word (obtained by dividing the total number of documents by the number of documents containing the term, and then taking the logarithm of that quotient):

$$idf(w) = log (\frac{number\ of\ documents}{num\ of\ documents\ containing\ w})$$

tf-idf is the product of term frequency and inverse document frequency, or tf * idf. 


```python
from sklearn.feature_extraction.text import TfidfVectorizer
tf_vec = TfidfVectorizer(token_pattern=r"([a-zA-Z]+(?:'[a-z]+)?)", stop_words=custom_sw)
X = tf_vec.fit_transform(corpus.body)

df = pd.DataFrame(X.toarray(), columns = tf_vec.get_feature_names())
df.head()
```


```python
corpus.iloc[313].body
```


```python
df.iloc[313].sort_values(ascending=False)[:10]
```

Let's compare the tfidf to the count vectorizer output for one document.


```python
vec = CountVectorizer(token_pattern=r"([a-zA-Z]+(?:'[a-z]+)?)", stop_words=custom_sw)
X = vec.fit_transform(corpus.body)

df_cv = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
df_cv
```


```python
df_cv.iloc[313].sort_values(ascending=False)[:10]
```

The tfidf lessoned the importance of some of the more common words, including a stopword which "also" which didn't make it into the stopword list.

It also assigns "nerds" more weight than power.  


```python
print(f'Nerds only shows up in document 313: {len(df_cv[df.nerds!=0])}')
print(f'Power shows up in {len(df_cv[df.power!=0])}')
```


```python
tf_vec.vocabulary_
```

# Pair: 

For a final exercise, work through in pairs the following exercise.

Create a document term matrix of the 1000 document corpus.  The vocabulary should have no stopwords, no numbers, no punctuation, and be lemmatized.  The Document-Term Matrix should be created using tfidf.
