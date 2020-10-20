
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

We will be working with a dataset which includes both **satirical** (The Onion) and real news (Reuters) articles. 

We refer to the entire set of articles as the **corpus**.  


```python
import pandas as pd

corpus = pd.read_csv('data/satire_nosatire.csv')

```

Our goal is to detect satire, so our target class of 1 is associated with The Onion articles.  

![the_onion](img/the_onion.jpeg) ![reuters](img/reuters.png)

Each article in the corpus is refered to as a **document**.


```python
corpus.shape
```




    (1000, 2)




```python
# It is a balanced dataset with 500 documents of each category. 
corpus.target.value_counts()
```




    1    500
    0    500
    Name: target, dtype: int64




```python
# Let's look at some example texts from both categories
corpus[corpus.target==1].sample().body.values

corpus[corpus.target==0].sample().body.values
```




    array([' A powerful Iranian council approved an anti-money laundering bill on Saturday, state media reported, a major step towards reforms that would bring Iran into line with global norms and could facilitate foreign trade in the face of U.S sanctions. Iran has been trying to implement standards set by the Financial Action Task Force (FATF), an inter-governmental organisation that underpins the fight against money laundering and terrorist financing. Foreign businesses say Iran’s compliance with FATF standards and its removal from the organisation’s blacklist are essential if they are to increase investment, especially after reimposition of the U.S. sanctions on Tehran. However, Iranian hardliners have opposed passing legislation toward compliance with the FATF, arguing it could hamper Iranian financial support for allies such as Lebanon’s Hezbollah, which the United States lists as a terrorist organisation. Parliament last year passed the anti-money laundering bill, one of four amendments Iran needs to implement to meet FATF requirements, but the Guardian Council, a vetting body, rejected it, saying it was against Islam and the constitution. On Saturday, the Expediency Council, a body intended to resolve disputes between parliament and the Guardian Council, approved the bill with some changes, state news agency IRNA said, quoting a member of the council. The move came after Ayatollah Sadeq Amoli Larijani - the chief of hardline judiciary - was appointed last week as the head of the Expediency Council. He is the brother of Ali Larijani, the speaker of the parliament. Seven months after his harsh dismissal of parliamentary efforts to adapt FATF and other international conventions on money laundering, Supreme Leader Ayatollah Ali Khamenei seems to have warmed to the reforms, a reversal that experts say is aimed at preventing Iran’s economic collapse. In recent months, cities have been rocked by demonstrations as factory workers, teachers, truck drivers and farmers protested against economic hardship. The sanctions have depressed the value of Iran’s rial currency and aggravated annual inflation fourfold to nearly 40 percent in November. U.S. President Donald Trump withdrew from a nuclear deal with Iran last year and reimposed the sanctions on its banking and energy sectors, hoping to curb its missile and nuclear programmes and counter its growing influence in the Middle East. European signatories of the nuclear deal are still committed to the accord and seek to launch a mechanism, a so-called special purpose vehicle (SPV), aiming to sidestep the U.S. financial system by using an EU intermediary to handle trade with Iran. The director general of Iran’s Strategic Council on Foreign Relations, an advisory body set up by Khamenei, voiced his support for the FATF-related bills on Friday. “It is better to finalise the FATF and CFT (counter financing of terrorism regimes) in the earliest time, so the Europeans have no excuse not to implement the (SPV) mechanism,” Abdolreza Faraji was quoted by semi-official ISNA new agency.'],
          dtype=object)



Let's think about our types of error and the use cases of being able to correctly separate satirical from authentic news. What type of error should we decide to optimize our models for?  


```python
X = corpus.body
y = corpus.target

```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=.2)
X_train
```




    29     “Did you see those golden globes?” an excited ...
    535     The most senior bankers to face criminal char...
    695     Five Louisiana children travelling to Disney ...
    557     A U.S. guided-missile destroyer sailed near d...
    836     U.S. commanders planning for the withdrawal o...
                                 ...                        
    106    The visiting contingent of communist Chinese c...
    270    Manufacturers of the newly unveiled robot call...
    860     Britain’s Royal Mail apologised on Friday aft...
    435    Kim Jong-Un set up an amazing fireworks show f...
    102    Thanks to the constant lies and closed curtain...
    Name: body, Length: 800, dtype: object




```python
from sklearn.feature_extraction.text import TfidfVectorizer

# A new preprocessing tool!
tfidf = TfidfVectorizer(token_pattern=r"([a-zA-Z]+(?:'[a-z]+)?)")

# Like always, we are fitting only on the training set
X_t_vec = tfidf.fit_transform(X_t)
# Here is our new dataframe of predictors
X_t_vec = pd.DataFrame(X_t_vec.toarray(), columns = tfidf.get_feature_names())
X_t_vec.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>aaaaaaah</th>
      <th>aaaaaah</th>
      <th>aaaaargh</th>
      <th>aaargh</th>
      <th>aah</th>
      <th>aahing</th>
      <th>aap</th>
      <th>aapl</th>
      <th>aaron</th>
      <th>...</th>
      <th>zooming</th>
      <th>zoos</th>
      <th>zor</th>
      <th>zozovitch</th>
      <th>zte</th>
      <th>zuckerberg</th>
      <th>zuercher</th>
      <th>zverev</th>
      <th>zych</th>
      <th>zzouss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.118115</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.062207</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.081322</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.190319</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.104993</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 19118 columns</p>
</div>



### We can push this data into any of our classification models



```python
from sklearn.linear_model import LogisticRegression
log_r = LogisticRegression()
log_r.fit(X_t_vec, y_t)

# Perfect on the training set. 
log_r.score(X_t_vec, y_t)
```




    1.0



# We proceed as usual:
    - Transform the validation set
    - Score the validation set
    - Plot a confusion matrix to check out the performance on different types of errors
 


```python
# Transform X_val
X_val_vec = tfidf.transform(X_val)
y_hat = log_r.predict(X_val_vec)
# Score val

print(log_r.score(X_val_vec, y_val))

# Plot confusion matrix
from sklearn.metrics import plot_confusion_matrix
# we should have a lot of true positive and true negatives
plot_confusion_matrix(log_r, X_val_vec, y_val)
```

    0.96875





    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x1a456f06a0>




![png](index_files/index_19_2.png)



```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

tfidf = TfidfVectorizer(token_pattern=r"([a-zA-Z]+(?:'[a-z]+)?)")
lr = LogisticRegression()
pipeline = make_pipeline(tfidf, lr)
pipeline
```




    Pipeline(memory=None,
             steps=[('tfidfvectorizer',
                     TfidfVectorizer(analyzer='word', binary=False,
                                     decode_error='strict',
                                     dtype=<class 'numpy.float64'>,
                                     encoding='utf-8', input='content',
                                     lowercase=True, max_df=1.0, max_features=None,
                                     min_df=1, ngram_range=(1, 1), norm='l2',
                                     preprocessor=None, smooth_idf=True,
                                     stop_words=None, strip_accents=None,
                                     sublinear_tf=False,
                                     token...ern="([a-zA-Z]+(?:'[a-z]+)?)",
                                     tokenizer=None, use_idf=True,
                                     vocabulary=None)),
                    ('logisticregression',
                     LogisticRegression(C=1.0, class_weight=None, dual=False,
                                        fit_intercept=True, intercept_scaling=1,
                                        l1_ratio=None, max_iter=100,
                                        multi_class='auto', n_jobs=None,
                                        penalty='l2', random_state=None,
                                        solver='lbfgs', tol=0.0001, verbose=0,
                                        warm_start=False))],
             verbose=False)




```python
cross_val_score(pipeline, X_train, y_train).mean()

```




    0.9737499999999999




```python

pipeline.fit(X_train, y_train)

```




    Pipeline(memory=None,
             steps=[('tfidfvectorizer',
                     TfidfVectorizer(analyzer='word', binary=False,
                                     decode_error='strict',
                                     dtype=<class 'numpy.float64'>,
                                     encoding='utf-8', input='content',
                                     lowercase=True, max_df=1.0, max_features=None,
                                     min_df=1, ngram_range=(1, 1), norm='l2',
                                     preprocessor=None, smooth_idf=True,
                                     stop_words=None, strip_accents=None,
                                     sublinear_tf=False,
                                     token...ern="([a-zA-Z]+(?:'[a-z]+)?)",
                                     tokenizer=None, use_idf=True,
                                     vocabulary=None)),
                    ('logisticregression',
                     LogisticRegression(C=1.0, class_weight=None, dual=False,
                                        fit_intercept=True, intercept_scaling=1,
                                        l1_ratio=None, max_iter=100,
                                        multi_class='auto', n_jobs=None,
                                        penalty='l2', random_state=None,
                                        solver='lbfgs', tol=0.0001, verbose=0,
                                        warm_start=False))],
             verbose=False)




```python

pipeline.score(X_test, y_test)
y_hat = pipeline.predict(X_test)
```


```python

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(pipeline, X_test, y_test)
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x1a44a16b70>




![png](index_files/index_24_1.png)


# 3 Preprocessing

### Tokenization 

In order to convert the texts into data suitable for machine learning, we need to break down the documents into smaller parts. 

The first step in doing that is **tokenization**.

Tokenization is the process of splitting documents into units of observations. We usually represent the tokens as __n-gram__, where n represent the consecutive words occuring in a document. In the case of unigram (one word token), the sentence "David works here" can be tokenized into?

"David", "works", "here"
"David works", "works here"

Let's consider the first document in our corpus:

There are many ways to tokenize our document. 

It is a long string, so the first way we might consider is to split it by spaces.


```python
first_document.split()[:30]
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
excessive = sentence_one.split(' ')[0]
Excessive = sentence_two.split(' ')[-2]
print(excessive, Excessive)
excessive == Excessive
```

    Excessive excessive





    False



### Let's fill in the list comprehension below to manually and remove capitals from the 1st document


```python
## Manual removal of capitals

manual_cleanup = [token.lower() for token in first_document.split(' ')]
manual_cleanup[:25]
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
     'weeks,']



## Punctuation

Like capitals, splitting on white space will create tokens which include punctuation that will muck up our semantics.  

Returning to the above example, 'gerrymandering' and 'gerrymandering.' will be treated as different tokens.

# Different ways to strip punctuation


```python
# Code here
no_punc = [word.translate(word.maketrans("","", punctuation)) for word in manual_cleanup]
no_punc
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
     'jesus',
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
     'this',
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
pattern = r"[a-zA-Z]+"

manual_cleanup = [re.search(pattern, word).group(0) for word in manual_cleanup if re.search(pattern, word)]
```


```python

# Better solution 
import re

pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
[re.match(regex_pattern, word).group(0) for word in manual_cleanup]
```




    ['noting',
     'resignation',
     'james',
     'mattis',
     'secretary',
     'defense',
     'marked',
     'ouster',
     'third',
     'top',
     'administration',
     'official',
     'less',
     'three',
     'weeks',
     'worried',
     'populace',
     'told',
     'reporters',
     'friday',
     'unsure',
     'many',
     'former',
     'trump',
     'staffers',
     'could',
     'safely',
     'reabsorb',
     'jesus',
     'take',
     'back',
     'assholes',
     'need',
     'time',
     'process',
     'one',
     'get',
     'next',
     'said',
     'yearold',
     'gregory',
     'birch',
     'naperville',
     'il',
     'echoing',
     'concerns',
     'million',
     'americans',
     'also',
     'noting',
     'country',
     'truly',
     'beginning',
     'reintegrate',
     'former',
     'national',
     'security',
     'advisor',
     'michael',
     'flynn',
     'sustainable',
     'handle',
     'maybe',
     'one',
     'two',
     'former',
     'members',
     'trump',
     'inner',
     'circle',
     'remainder',
     'year',
     'country',
     'limits',
     'us',
     'populace',
     'confirmed',
     'could',
     'handle',
     'pieces',
     'shit',
     'trying',
     'rejoin',
     'society']



### Stopwords

Stopwords are the **filler** words in a language: prepositions, articles, conjunctions. They have low semantic value, and almost always need to be removed.  

Luckily, NLTK has lists of stopwords ready for our use.

Let's see which stopwords are present in our first document.

Let's also use the **FreqDist** tool to look at the makeup of our text before and after removal


```python
custom_sw.extend(['could', 'one'])
manual_cleanup = [token for token in manual_cleanup if token not in custom_sw]

fdist = FreqDist(manual_cleanup)
plt.figure(figsize=(10,10))
fdist.plot(30)
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
import re

# 1 or more digits
pattern = '\d+'

number = re.findall( pattern, first_document)
number
```

Sklearn and NLTK provide us with a suite of **tokenizers** for our text preprocessing convenience.

# Stemming

Most of the semantic meaning of a word is held in the root, which is usually the beginning of a word.  Conjugations and plurality do not change the semantic meaning. "eat", "eats", and "eating" all have essentially the same meaning packed into eat.   

Stemmers consolidate similar words by chopping off the ends of the words.

![stemmer](img/stemmer.png)

There are different stemmers available.  The two we will use here are the **Porter** and **Snowball** stemmers.  A main difference between the two is how agressively it stems, Porter being less agressive.

# Lemming

Lemming is a bit more sophisticated that the stem choppers.  Lemming uses part of speech tagging to determine how to transform a word.  In that 
Lemmatization returns real words. For example, instead of returning "movi" like Porter stemmer would, "movie" will be returned by the lemmatizer.

- Unlike Stemming, Lemmatization reduces the inflected words properly ensuring that the root word belongs to the language.  It can handle words such as "mouse", whose plural "mice" the stemmers would not lump together with the original. 

- In Lemmatization, the root word is called Lemma. 

- A lemma (plural lemmas or lemmata) is the canonical form, dictionary form, or citation form of a set of words.

![lemmer](img/lemmer.png)


Lemmatizers depend on POS tagging, and defaults to noun.

With a little bit of work, we can POS tag our text.

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

That is not very exciting for one document. The idea is to make a document term matrix for all of the words in our corpus.

Our document term matrix gets bigger and bigger, with more and more zeros, becoming sparser and sparser.

We can set upper and lower limits to the word frequency.

### TF-IDF 
There are many schemas for determining the values of each entry in a document term matrix, and one of the most common schema is called the TF-IDF -- term frequency-inverse document frequency. Essentially, tf-idf *normalizes* the raw count of the document term matrix. And it represents how important a word is in the given document. 

> The goal of using tf-idf instead of the raw frequencies of occurrence of a token in a given document is to scale down the impact of tokens that occur very frequently in a given corpus and that are hence empirically less informative than features that occur in a small fraction of the training corpus.

- TF (Term Frequency)
term frequency is the frequency of the word in the document divided by the total words in the document.

- IDF (inverse document frequency)
IDF represents the measure of how much information the word provides, i.e., if it's common or rare across all documents. It is the logarithmically scaled inverse fraction of the documents that contain the word (obtained by dividing the total number of documents by the number of documents containing the term, and then taking the logarithm of that quotient):

$$idf(w) = log (\frac{number\ of\ documents}{num\ of\ documents\ containing\ w})$$

tf-idf is the product of term frequency and inverse document frequency, or tf * idf. 

Let's compare the tfidf to the count vectorizer output for one document.

The tfidf lessoned the importance of some of the more common words, including a stopword which "also" which didn't make it into the stopword list.

It also assigns "nerds" more weight than power.  

# Pair: 

For a final exercise, work through in pairs the following exercise.

Create a document term matrix of the 1000 document corpus.  The vocabulary should have no stopwords, no numbers, no punctuation, and be lemmatized.  The Document-Term Matrix should be created using tfidf.


```python
corpus = pd.read_csv('data/satire_nosatire.csv')

```


```python
def doc_preparer(doc, stop_words=custom_sw):
    '''
    
    :param doc: a document from the satire corpus 
    :return: a document string with words which have been 
            lemmatized, 
            parsed for stopwords, 
            made lowercase,
            and stripped of punctuation and numbers.
    '''
    
    regex_token = RegexpTokenizer(r"([a-zA-Z]+(?:’[a-z]+)?)")
    doc = regex_token.tokenize(doc)
    doc = [word.lower() for word in doc]
    doc = [word for word in doc if word not in stop_words]
    doc = pos_tag(doc)
    doc = [(word[0], get_wordnet_pos(word[1])) for word in doc]
    lemmatizer = WordNetLemmatizer() 
    doc = [lemmatizer.lemmatize(word[0], word[1]) for word in doc]
    return ' '.join(doc)

```


```python
docs = [doc_preparer(doc) for doc in corpus.body]
```


```python
tf_idf = TfidfVectorizer(min_df = .05)
X = tf_idf.fit_transform(docs)

df = pd.DataFrame(X.toarray())
df.columns = tf_idf.vocabulary_
df.head()
```
