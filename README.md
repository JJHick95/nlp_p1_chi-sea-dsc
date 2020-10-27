
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
from src.student_caller import one_random_student
from src.student_list import student_first_names
```


```python
"Name 3 algorithms whose effectiveness depends on scaling"
one_random_student(student_first_names)
```


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

## 1. Overview of NLP
NLP allows computers to interact with text data in a structured and sensible way.  We will be breaking up series of texts into individual words (or groups of words), and isolating the words with **semantic value**.  We will then compare texts with similar distributions of these words, and group them together.

In this section, we will discuss some steps and approaches to common data analytic procedures for text to learn how computers can understand human language, its meaning and sentiments. Some of the applications of natural language processing are:

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

You have now been introduced to a suite of concepts concerning how to transform data of different raw forms into forms that computers can learn from. We have learned how to scale, one hot encode, bin, and resample continuous and categorical independent features for regression and classification tasks.  We have learned how to arrange autocorrelated series chronologically to predict with past value.  And now we will learn to break up text into numerical dataframes that reflect semantic value.


```python
#!pip install nltk
# conda install -c anaconda nltk
```

We will be working with a dataset which includes both **satirical** (The Onion) and serious news (Reuters) articles. 

### Vocab
> We refer to the entire set of articles as the **corpus**.  


```python
# Let's import our corpus. 
import pandas as pd

```

Our goal is to detect satire, so our target class of 1 is associated with The Onion articles. 

![the_onion](img/the_onion.jpeg) ![reuters](img/reuters.png)

This is a highly relavent task.  If we could separate real news from fictitious news, we would be able to potentially flag the latter as such.  This does come with risks.  If we deploy a model which mislabel real news as ficticious news, we will open ourselves to all sorts of criticism.  A false positive in this sense to bury a story or do damage to a reporter's reputation. 

#### More Vocab
> Each article in the corpus is refered to as a **document**.


```python
# How many documents are there in the corpus?
```


```python
# What is the class balance?
```


```python
# Let's look at some example texts from both categories.

# Category 1
corpus[corpus.target==1].sample(random_state=2).body.values
```


```python
# Category 0
corpus[corpus.target==0].sample(random_state=1).body.values
```

Let's think about our types of error and the use cases of being able to correctly separate satirical from authentic news. What type of error should we decide to optimize our models for?  


```python
# Help me fill in the blanks:

# Pass in the body of the documents as raw text
X = None
y = None
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.2)
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

X_t_vec.head()
```


```python
# That's a lot of columns
X_t_vec.shape
```

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

# score val
print()

# Plot confusion matrix

```

#### How did your model do?  

Probably well.  
It's pretty amazing the patterns, even a relatively low power one such as logistic regression, can find in text data.

# 3 Preprocessing

### Tokenization 

In order to convert the texts into data suitable for machine learning, we need to break down the documents into smaller parts. 

The first step in doing that is **tokenization**.

Tokenization is the process of splitting documents into units of observations. We usually represent the tokens as __n-gram__, where n represent the consecutive words occuring in a document. In the case of unigram (one word token), the sentence "David works here" can be tokenized into?

"David", "works", "here"
"David works", "works here"

Let's consider the first document in our corpus:


```python
first_document = None
```

There are many ways to tokenize our document. 

It is a long string, so the first way we might consider is to split it by spaces.


```python
# code
```

We are attempting to create a list of tokens whose semantic value reflects the class the document is associated with.  What issues might you see with the list of tokens generated by a simple split as performed above?


```python
one_random_student(student_first_names)
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

### Let's fill in the list comprehension below to manually and remove capitals from the 1st document


```python
manual_cleanup = None
```


```python
print(f"Our initial token set for our first document is {len(manual_cleanup)} words long")
```


```python
print(f"Our initial token set for our first document has {len(set(first_document.split()))} unique words")
```


```python
print(f"After remove caps, our first document has {len(set(manual_cleanup))} unique words")
```

By removing capitals, we decrease the total unique word count in our first document by 2.  

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


```python
no_punc = [word.translate(word.maketrans("","", punctuation)) for word in manual_cleanup]

no_punc
```

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

Other helpful resources:

https://regexcrossword.com/  
https://www.regular-expressions.info/tutorial.html


```python
import re
# Test out a word and search it

# Create a pattern that matches only letters to strip punctuation


# Use re.search to search for a word.
target_word = manual_cleanup[10]

# Use the .group() method to return the matched word


```


```python
# Use a list comprehension to search each word in the word list and return the match.
# Use an if statement to make sure the .group() method does not throw an error.

# Code here
```


```python
print(f"After removing punctuation, our first document has {len(set(manual_cleanup))} unique words")
```

### Stopwords

Stopwords are the **filler** words in a language: prepositions, articles, conjunctions. They have low semantic value, and almost always need to be removed.  

Luckily, NLTK has lists of stopwords ready for our use.


```python
from nltk.corpus import stopwords
stopwords.words('english')[:10]

# Try other langauges below
```

As you notice above, stopwords come from the [nltk corpus package](http://www.nltk.org/book_1ed/ch02.html#tab-corpora).  The corpus package contains a variety of texts that are free to download and use.

Let's see which stopwords are present in our first document.


```python
stops = [token for token in manual_cleanup if token in stopwords.words('english')]

```


```python
stops = [token for token in manual_cleanup if token in stopwords.words('english')]
stops[:10]
```


```python
print(f'There are {len(stops)} stopwords in the first document')
```


```python
print(f'That is {len(stops)/len(manual_cleanup): .2%} of our text')
```

Let's also use the **FreqDist** tool to look at the makeup of our text before and after removal


```python
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
```


```python
manual_cleanup = [token for token in manual_cleanup if token not in stopwords.words('english')]
```


```python
# We can also customize our stopwords list
```


```python
manual_cleanup = [token for token in manual_cleanup if token not in custom_sw]

```


```python
print(f'After removing stopwords, there are {len(set(manual_cleanup))} unique words left')
```


```python
fdist = FreqDist(manual_cleanup)
plt.figure(figsize=(10,10))
fdist.plot(30)
```


```python
# From the Frequency Dist plot, there are perhaps some more stopwords we can remove

```

#### Numbers

Numbers also usually have low semantic value. Their removal can help improve our models. 

To remove them, we will use regular expressions, a powerful tool which you may already have some familiarity with.

Bringing back our regex symbols, we can quickly figure out patterns which will filter out numeric values.

A few key symbols:
  - . : matches any character
  - \d, \w, \s : represent digit, word, whitespace  
  - *, ?, +: matches 0 or more, 0 or 1, 1 or more of the preceding character  
  - [A-Z]: matches any capital letter  
  - [a-z]: matches lowercase letter  


```python
# pattern to remove numbers (probably same as above)

no_num_pattern = None

test_string = "Reno 911"

re.search(no_num_pattern, test_string).group()

```

# Pair Exercise:  
Sklearn and NLTK provide us with a suite of **tokenizers** for our text preprocessing convenience.



```python
import nltk
from nltk.tokenize import regexp_tokenize, word_tokenize, RegexpTokenizer

```

We can use them to condense the steps above.  As we will see, even more steps will be condensed in the vectorizors introduced at the end of the notebook.  

For now, we will still need to make our list lowercase and remove stopwords by hand. 

It is important to get used to the process of tokenizing by hand, since it will give us more freedom in certain preprocessing steps (see stemmers/lemmers below).

For this exercise, take the sample_doc below and use the tokenizer of your choice to create word tokens.  If you use the regexp vectorizor, I have included a regex pattern that does not exclude contractions.  Feel free to pass that in as an argument.

After tokenizing, make tokens lowercase, and remove stopwords.


```python
sample_doc = X_t.sample(random_state=42).values[0]
sample_doc
```


```python
import re
re.findall(r"([a-zA-Z]+(?:'[a-z]+)?)" , "I'd")
```


```python
# your code here

# Use the tokenizer of your choice using the .tokenize() method.
tokenizer = None

# make lowercase

# remove stopwords


```

# Stemming

Most of the semantic meaning of a word is held in the root, which is usually the beginning of a word.  Conjugations and plurality do not change the semantic meaning. "eat", "eats", and "eating" all have essentially the same meaning packed into eat.   

Stemmers consolidate similar words by chopping off the ends of the words.

![stemmer](img/stemmer.png)

There are different stemmers available.  The two we will use here are the **Porter** and **Snowball** stemmers.  A main difference between the two is how agressively it stems, Porter being less agressive.


```python
from nltk.stem import *

# instantiate a PorterStemmer and a SnowballStemmer.  Pass language='english' as an argument to the Snowball
```


```python
# using the .stem() method, try out stemming some words from the sample doc

```


```python
difference_count = 0
for word in sample_doc:
    p_word = p_stemmer.stem(word)
    s_word = s_stemmer.stem(word)
    
    if p_word != s_word:
        print(p_word, s_word)
        difference_count += 1
        
print("\n" f"Of the {len(sample_doc)} words in the sample doc, only {difference_count} are stemmed differently")
    
```


```python
# Let's use the snowball stemmer, and stem all the words in the doc

```


```python
fdist = FreqDist(sample_doc)
plt.figure(figsize=(10,10))
fdist.plot(30)
```


```python
print(f'Stemming reduced our token count {len(set(sample_doc))} to unique tokens')
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
  
# instantiate a lemmatizer
lemmatizer = None
```


```python
# think of a noun with an irregular plural form and pass the string as an argument to the lemmatizer
print(f'<Irregular noun> becomes: {lemmatizer.lemmatize()}')

```


```python
new_sample = X_t.sample().values[0]
tokenizer = RegexpTokenizer(r"([a-zA-Z]+(?:[’'][a-z]+)?)")

new_sample = tokenizer.tokenize(new_sample)
new_sample
```


```python
print(f'{new_sample[8]} becomes: {lemmatizer.lemmatize(new_sample[8])}')
```

Lemmatizers depend on POS tagging, and defaults to noun.

With a little bit of work, we can POS tag our text.


```python
pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
tokenizer = RegexpTokenizer(r"([a-zA-Z]+(?:[’'][a-z]+)?)")
new_sample = [token for token in new_sample if token not in custom_sw]
new_sample
```


```python
nltk.download('tagsets')
nltk.help.upenn_tagset()
```


```python
from nltk import pos_tag
# Use nltk's pos_tag to tag our words
new_sample_tagged = None
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
new_sample_tagged = [(token[0], get_wordnet_pos(token[1]))
             for token in new_sample_tagged]
```


```python
# Now lemmatize with the POS fed into the lemmatizer

```


```python
new_sample_lemmed[:10]
```


```python
print(f'There are {len(set(new_sample_lemmed))} unique lemmas')
```


```python
fdist = FreqDist(new_sample_lemmed)
plt.figure(figsize=(10,10))
fdist.plot(30)
```

## 4. Feature Engineering for NLP 
In order to pass text data to machine learning algorithm and perform classification, we need to represent the features in a sensible way. One such method is called **Bag-of-words (BoW)**. 

A bag-of-words model, or BoW for short, is a way of extracting **features** from text for use in modeling. A bag-of-words is a representation of text that describes the occurrence of words within a document. It involves two things:

- A vocabulary of known words.
- A measure of the presence of known words.

It is called a “bag” of words, because any information about the order or structure of words in the document is discarded. The model is only concerned with whether words occur in the document, **not where** in the document. The intuition behind BoW is that a document is similar to another if they have similar contents. Bag of Words method can be represented as **Document Term Matrix**, or Term Document Matrix, in which each column is an unique vocabulary, each observation is a document. For example:

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

# instantiate a count vectorizer
vec = None

# fit vectorizor on our lemmed sample. Note, the vectorizer takes in raw texts, so we need to join all of our lemmed tokens.
X = None
```

That is not very exciting for one document. The idea is to make a document term matrix for all of the words in our corpus.

We can pass in arguments such as a regex pattern, a list of stopwords, and an ngram range to do our preprocessing in one fell swoop.   
*Note lowercase defaults to true.*


```python
# pass in the regex from above, our custom stopwords, and an ngram range: [1,1], [1,2] , [1,3]
```

Our document term matrix gets bigger and bigger, with more and more zeros, becoming sparser and sparser.

> In case you forgot, a sparse matrix "is a matrix in which most of the elements are zero." [wikipedia](https://en.wikipedia.org/wiki/Sparse_matrix)

We can set upper and lower limits to the word frequency.


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


```python

```
