
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

    Prabhakar



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

### Vocab
> We refer to the entire set of articles as the **corpus**.  


```python
# Let's import our corpus. 
import pandas as pd

```


```python
#__SOLUTION__
import pandas as pd

corpus = pd.read_csv('data/satire_nosatire.csv')

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
#__SOLUTION__
corpus.shape
```




    (1000, 2)




```python
# What is the class balance?
```


```python
#__SOLUTION__
# It is a balanced dataset with 500 documents of each category. 
corpus.target.value_counts()
```




    1    500
    0    500
    Name: target, dtype: int64




```python
# Let's look at some example texts from both categories.

# Category 1

```


```python
# Category 0
```


```python
#__SOLUTION__
# Let's look at some example texts from both categories
corpus[corpus.target==1].sample().body.values

corpus[corpus.target==0].sample().body.values
```




    array([' Gabon foiled an attempted military coup on Monday, killing two suspected plotters and capturing seven others just hours after they took over state radio in a bid to end 50 years of rule by President Ali Bongo’s family. Government spokesman Guy-Bertrand Mapangou announced the deaths and arrests after soldiers briefly seized the radio station and broadcast a message saying Bongo was no longer fit for office after suffering a stroke in Saudi Arabia in October. The quick failure of Monday’s coup and the lack of widespread support suggest further efforts to overthrow Bongo are unlikely, analysts said. But the attempt alone shows a growing frustration with a government weakened by the President’s secretive medical leave. On Dec. 31, in one of his first television appearances since the stroke, Bongo, 59, slurred his speech and he appeared unable to move his right arm. It is unclear if he is able to walk. He has been in Morocco since November to continue treatment. In a radio message at 4:30 a.m. (0330 GMT), Lieutenant Kelly Ondo Obiang, who described himself as an officer in the Republican Guard, said Bongo’s New Year’s Eve address “reinforced doubts about the president’s ability to continue to carry out of the responsibilities of his office”. Outside the radio station, loyalist soldiers fired teargas to disperse about 300 people who had come out into the streets to support the coup attempt, a Reuters witness said. Helicopters circled overhead and there was a strong military and police presence on the streets. Most of the beachside capital was quiet, however, and a government spokesman said the situation was under control after the arrests. Residents said Internet access was cut. “The government is in place. The institutions are in place,” Mapangou told France 24. The Bongo family has ruled the oil-producing country since 1967. Bongo has been president since succeeding his father, Omar, who died in 2009. His re-election in 2016 was marred by claims of fraud and violent protest. The economy was long buoyed by oil revenues, much of which went to a moneyed elite while most of the two-million population live in deep poverty. In Libreville, expensive western hotels overlook the Atlantic Ocean to the west and the capital’s hillside shanties to the east. A sharp drop in oil output and prices in recent years has squeezed revenues, raised debt and stoked discontent. Oil workers’ strikes have become more common. Economic growth was 2 percent last year, down from over 7 percent in 2011. The coup indicates “broad socio-economic and political frustration with Gabon’s leadership, which has been weakened by the suspected incapacitation of its strongman president,” Exx Africa Business Risk Intelligence said in a report. The international community condemned the coup attempt, including former colonial ruler France which urged its 8,900 citizens registered in Gabon to avoid moving around Libreville. “Gabon’s stability can only be ensured in strict compliance with the provisions of its constitution,” French foreign ministry spokeswoman Agnes von der Muhll said. African Union Commission Chairman Moussa Faki Mahamat reaffirmed “the AU’s total rejection of all unconstitutional changes of power.” In a video on social media, Ondo is seen in a radio studio wearing military fatigues and a green beret as he reads the statement. Two other soldiers with rifles stand behind him. Ondo said the coup attempt was by a group called the Patriotic Movement of the Defence and Security Forces of Gabon against “those who, in a cowardly way, assassinated our young compatriots on the night of August 31, 2016,” a reference to violence after Bongo was declared winner of a disputed election. Bongo won the poll by fewer than 6,000 votes, sparking deadly clashes between protesters and police during which the parliament was torched. “President Bongo’s record as defence minister under his father lowers the possibility that current military leadership is supportive of his ouster,” said Judd Devermont of the Center for Strategic and International Studies in Washington. France has a permanent force of 300 soldiers in Gabon. The United States also sent about 80 soldiers to Gabon last week in response to possible violence in Democratic Republic of Congo after a presidential election there. Foreign governments have often suspected Bongo and members of his government of corruption, accusations they have denied. During his father’s rule, Gabon was a pillar of “La Francafrique”, a web of influence that gave French companies favoured access to African autocrats. Gabon’s dollar-denominated sovereign debt <XS1003557870=TE > <US362420AC51=TE > tumbled in early trading, with both outstanding bonds losing around 3 cents in the dollar. However, prices recovered in late morning, with bonds trading around half a cent lower. Additional reporting by David Lewis, Maggie Fick, Ange Aboa and Karin Strohecker; Writing by Aaron Ross and Edward McAllister; Editing by Simon Cameron-Moore, Raissa Kasolowsky, William Maclean MORE FROM REUTERS SPONSORED SPONSORED All quotes delayed a minimum of 15 minutes. See here for a complete list of exchanges and delays. © 2019 Reuters. All Rights Reserved.'],
          dtype=object)



Let's think about our types of error and the use cases of being able to correctly separate satirical from authentic news. What type of error should we decide to optimize our models for?  


```python
# Help me fill in the blanks:

# Pass in the body of the documents as raw text
X = None
y = None
```


```python
#__SOLUTION__
X = corpus.body
y = corpus.target

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


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-23-216d598a5d82> in <module>
          8 
          9 # Here is our new dataframe of predictors
    ---> 10 X_t_vec = pd.DataFrame(X_t_vec.toarray(), columns = tfidf.get_feature_names())
         11 
         12 X_t_vec.head()


    AttributeError: 'NoneType' object has no attribute 'toarray'



```python
#__SOLUTION__
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# A new preprocessing tool!

# This tool removes case punctuation, numbers, and stopwords in one fell swoop.
# We will break this down below
tfidf = TfidfVectorizer(token_pattern=r"([a-zA-Z]+(?:'[a-z]+)?)", stop_words=stopwords.words('english'))

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
      <th>aaaaaaah</th>
      <th>aaaaaah</th>
      <th>aaaaargh</th>
      <th>aaargh</th>
      <th>aah</th>
      <th>aahing</th>
      <th>aap</th>
      <th>aapl</th>
      <th>aaron</th>
      <th>ab</th>
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
<p>5 rows × 18970 columns</p>
</div>




```python
# That's a lot of columns
X_t_vec.shape
```




    (640, 18970)



### We can push this data into any of our classification models



```python
# Code here
```


```python
#__SOLUTION__
from sklearn.linear_model import LogisticRegression
log_r = LogisticRegression()
log_r.fit(X_t_vec, y_t)

# Perfect on the training set. 
log_r.score(X_t_vec, y_t)
```




    0.996875



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

    



```python
#__SOLUTION__
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

    0.9875





    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x117a034a8>




![png](index_files/index_36_2.png)


#### How did your model do?  

Probably well.  
It's pretty amazing the patterns, even a relatively low power one such as logistic regression, can find in text data.


```python
#__SOLUTION__
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
#__SOLUTION__
cross_val_score(pipeline, X_train, y_train).mean()

```




    0.9737499999999999




```python
#__SOLUTION__

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
#__SOLUTION__

pipeline.score(X_test, y_test)
y_hat = pipeline.predict(X_test)
```


```python
#__SOLUTION__

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(pipeline, X_test, y_test)
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x117c6a710>




![png](index_files/index_42_1.png)


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


```python
#__SOLUTION__
first_document = corpus.iloc[0].body
```

There are many ways to tokenize our document. 

It is a long string, so the first way we might consider is to split it by spaces.


```python
# code
```


```python
#__SOLUTION__
first_document.split()[:30]
```




    ['Noting',
     'that',
     'the',
     'resignation',
     'of',
     'James',
     'Mattis',
     'as',
     'Secretary',
     'of',
     'Defense',
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
     'weeks,',
     'a',
     'worried',
     'populace',
     'told',
     'reporters']



We are attempting to create a list of tokens whose semantic value reflects the class the document is associated with.  What issues might you see with the list of tokens generated by a simple split as performed above?


```python
one_random_student(student_first_names)
```

    Sindhu


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
#__SOLUTION__
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


By removing capitals, we decrease the total unique word count in our first document by 2.  That may not seem like much, but across an entire corpus, it will make a big difference.

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

Other helpful resources:

https://regexcrossword.com/  
https://www.regular-expressions.info/tutorial.html


```python
import re
# Test out a word and search it

# Create a pattern that matches only letters to strip punctuation
pattern = None

# Use re.search to search for a word.
target_word = manual_cleanup[10]

# Use the .group() method to return the matched word


```


```python
#__SOLUTION__
pattern = r"[a-zA-Z]+"

target_word = manual_cleanup[10]
re.search(pattern, target_word).group(0)


```




    'defense'




```python
# Use a list comprehension to search each word in the word list and return the match.
# Use an if statement to make sure the .group() method does not throw an error.

# Code here
```


```python
# __SOLUTION__
manual_cleanup = [re.search(pattern, word).group(0) for word in manual_cleanup if re.search(pattern, word)]
```


```python
#__SOLUTION__

# Better solution 
import re

pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
[re.match(pattern, word).group(0) for word in manual_cleanup]
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
     'can',
     'just',
     'take',
     'back',
     'these',
     'assholes',
     'all',
     'at',
     'once',
     'need',
     'time',
     'to',
     'process',
     'one',
     'before',
     'we',
     'get',
     'the',
     'next',
     'said',
     'year',
     'gregory',
     'birch',
     'of',
     'naperville',
     'il',
     'echoing',
     'the',
     'concerns',
     'of',
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
     'i',
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
     'trump',
     'inner',
     'circle',
     'over',
     'the',
     'remainder',
     'of',
     'the',
     'year',
     'but',
     'that',
     'it',
     'this',
     'country',
     'has',
     'its',
     'limits',
     'the',
     'u',
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
print(f"After removing punctuation, our first document has {len(set(manual_cleanup))} unique words")
```

    After removing punctuation, our first document has 107 unique words


### Stopwords

Stopwords are the **filler** words in a language: prepositions, articles, conjunctions. They have low semantic value, and almost always need to be removed.  

Luckily, NLTK has lists of stopwords ready for our use.


```python
from nltk.corpus import stopwords
stopwords.words('english')[:10]

# Try other langauges below
```




    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]



As you notice above, stopwords come from the [nltk corpus package](http://www.nltk.org/book_1ed/ch02.html#tab-corpora).  The corpus package contains a variety of texts that are free to download and use.

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
#__SOLUTION__
fdist = FreqDist(manual_cleanup)
plt.figure(figsize=(10,10))
fdist.plot(30)

```


![png](index_files/index_91_0.png)





    <matplotlib.axes._subplots.AxesSubplot at 0x121250390>




```python
manual_cleanup = [token for token in manual_cleanup if token not in stopwords.words('english')]
```


```python
# We can also customize our stopwords list
```


```python
#__SOLUTION__
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

    After removing stopwords, there are 74 unique words left



```python
fdist = FreqDist(manual_cleanup)
plt.figure(figsize=(10,10))
fdist.plot(30)
```


![png](index_files/index_97_0.png)





    <matplotlib.axes._subplots.AxesSubplot at 0x1222c3b00>




```python
# From the Frequency Dist plot, there are perhaps some more stopwords we can remove

```


```python
#__SOLUTION__
custom_sw.extend(['could', 'one'])
manual_cleanup = [token for token in manual_cleanup if token not in custom_sw]

fdist = FreqDist(manual_cleanup)
plt.figure(figsize=(10,10))
fdist.plot(30)
```


![png](index_files/index_99_0.png)





    <matplotlib.axes._subplots.AxesSubplot at 0x1222d6470>



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


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-67-5a81b77bcf83> in <module>
          5 test_string = "Reno 911"
          6 
    ----> 7 re.search(no_num_pattern, test_string).group()
    

    ~/anaconda3/lib/python3.7/re.py in search(pattern, string, flags)
        181     """Scan through string looking for a match to the pattern, returning
        182     a Match object, or None if no match was found."""
    --> 183     return _compile(pattern, flags).search(string)
        184 
        185 def sub(pattern, repl, string, count=0, flags=0):


    ~/anaconda3/lib/python3.7/re.py in _compile(pattern, flags)
        283         return pattern
        284     if not sre_compile.isstring(pattern):
    --> 285         raise TypeError("first argument must be string or compiled pattern")
        286     p = sre_compile.compile(pattern, flags)
        287     if not (flags & DEBUG):


    TypeError: first argument must be string or compiled pattern



```python
#__SOLUTION__
no_num_pattern = r'[a-zA-Z]*'
test_string = "Reno 911"

re.search(no_num_pattern, test_string).group()
```




    'Reno'



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




    '“Yes, we can confirm that Mrs. May is actively looking for other employment opportunities. She has a copy of the Evening Standard where they post some job ads. In fact, there is one job she is applying to, and that is a shoe saleswoman in Barking, East London. This would be a great opportunity for the ex-PM to be around what she loves, shoes.” The final nail in the coffin for many was Theresa May approving that the European Courts of Justice would have full jurisdiction in Britain after Brexit. This is tantamount to Britain not leaving the EU at all. The soon-to-be ex-PM also conceded to pay Brussels whatever they want with a 100 billion euro payment, of course taken out of the purse of the British taxpayer. This blackmail ransom money could have easily been avoided by not pandering to the EU’s ridiculous demands and simply leaving the talks. Instead, Theresa May lifted up her skirt, bent over the table and was used like a piece of sallow rotting meat by EU eurocrats. In a failed Brexit, led by Remainers, what did you expect? So, what happens when Theresa May is shown the door? Yes, there could be another impromptu general election, which would no doubt bring in Comrade Corbyn. In a weird, anarchic sense, this would be a fitting end to a Brexit that is not really a Brexit but one only by name. We would naturally have anarchy in the streets, massive chaos in government, banking services completely disrupted and stripped of their privileges, and a royal family on the run, turfed out of their palaces by the people’s army and Momentum. What about the other Brexiteers in the Tory government? Well, according to many voters, they did not have the balls to stand up to Theresa May when she was in power, therefore they failed their country miserably. They may call themselves Brexiteers, however, they are, much like the fake Brexit we are being delivered, fake as well.'




```python
import re
re.findall(r"([a-zA-Z]+(?:'[a-z]+)?)" , "I'd")
```




    ["I'd"]




```python
# your code here

# Use the tokenizer of your choice using the .tokenize() method.
tokenizer = None

# make lowercase

# remove stopwords


```


```python
#__SOLUTION__

pattern = "([a-zA-Z]+(?:'[a-z]+)?)"

tokenizer = RegexpTokenizer(r"([a-zA-Z]+(?:[’'][a-z]+)?)")

sample_doc = X_t.sample(random_state=42).values[0]
sample_doc = tokenizer.tokenize(sample_doc)

sample_doc = [token.lower() for token in sample_doc]
sample_doc = [token for token in sample_doc if token not in custom_sw]
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-61-6e02bc632e8d> in <module>
          9 
         10 sample_doc = [token.lower() for token in sample_doc]
    ---> 11 sample_doc = [token for token in sample_doc if token not in custom_sw]
    

    <ipython-input-61-6e02bc632e8d> in <listcomp>(.0)
          9 
         10 sample_doc = [token.lower() for token in sample_doc]
    ---> 11 sample_doc = [token for token in sample_doc if token not in custom_sw]
    

    NameError: name 'custom_sw' is not defined


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
#__SOLUTION__
p_stemmer = PorterStemmer()
s_stemmer = SnowballStemmer(language="english")
```


```python
# using the .stem() method, try out stemming some words from the sample doc

```


```python
#__SOLUTION__
p_stemmer.stem(sample_doc[0])
s_stemmer.stem(sample_doc[1])
```




    'confirm'




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

    ye yes
    mr mrs
    eu’ eu
    ye yes
    gener general
    people’ peopl
    
    Of the 171 words in the sample doc, only 6 are stemmed differently



```python
# Let's use the snowball stemmer, and stem all the words in the doc

```


```python
#__SOLUTION__
sample_doc = [s_stemmer.stem(word) for word in sample_doc]
```


```python
fdist = FreqDist(sample_doc)
plt.figure(figsize=(10,10))
fdist.plot(30)
```


![png](index_files/index_123_0.png)





    <matplotlib.axes._subplots.AxesSubplot at 0x1a2d019c50>




```python
print(f'Stemming reduced our token count {len(set(sample_doc))} to unique tokens')
```

    Stemming reduced our token count 137 to unique tokens


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
#__SOLUTION__
lemmatizer = WordNetLemmatizer() 
```


```python
# think of a noun with an irregular plural form and pass the string as an argument to the lemmatizer
print(f'<Irregular noun> becomes: {lemmatizer.lemmatize()}')

```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-86-55794db2d676> in <module>
          1 # think of a noun with an irregular plural form and pass the string as an argument to the lemmatizer
    ----> 2 print(f'<Irregular noun> becomes: {lemmatizer.lemmatize()}')
    

    TypeError: lemmatize() missing 1 required positional argument: 'word'



```python
#__SOLUTION__
print(f'Mice becomes: {lemmatizer.lemmatize("mice")}')
```

    Mice becomes: mouse



```python
new_sample = X_t.sample().values[0]
tokenizer = RegexpTokenizer(r"([a-zA-Z]+(?:[’'][a-z]+)?)")

new_sample = tokenizer.tokenize(new_sample)
new_sample
```




    ['Sudanese',
     'authorities',
     'are',
     'blocking',
     'access',
     'to',
     'popular',
     'social',
     'media',
     'platforms',
     'used',
     'to',
     'organise',
     'and',
     'broadcast',
     'nationwide',
     'anti',
     'government',
     'protests',
     'triggered',
     'by',
     'an',
     'economic',
     'crisis',
     'internet',
     'users',
     'say',
     'Sudan',
     'has',
     'been',
     'rocked',
     'by',
     'near',
     'daily',
     'demonstrations',
     'over',
     'the',
     'past',
     'two',
     'weeks',
     'Protesters',
     'have',
     'set',
     'alight',
     'ruling',
     'party',
     'buildings',
     'and',
     'have',
     'called',
     'on',
     'President',
     'Omar',
     'al',
     'Bashir',
     'who',
     'took',
     'power',
     'in',
     'to',
     'step',
     'down',
     'In',
     'a',
     'country',
     'where',
     'the',
     'state',
     'tightly',
     'controls',
     'traditional',
     'media',
     'the',
     'internet',
     'has',
     'become',
     'a',
     'key',
     'information',
     'battleground',
     'Of',
     'Sudan’s',
     'million',
     'people',
     'some',
     'million',
     'use',
     'the',
     'internet',
     'and',
     'more',
     'than',
     'million',
     'own',
     'mobile',
     'phones',
     'local',
     'media',
     'say',
     'Authorities',
     'have',
     'not',
     'repeated',
     'the',
     'internet',
     'blackout',
     'they',
     'imposed',
     'during',
     'deadly',
     'protests',
     'in',
     'But',
     'the',
     'head',
     'of',
     'Sudan’s',
     'National',
     'Intelligence',
     'and',
     'Security',
     'Service',
     'Salah',
     'Abdallah',
     'told',
     'a',
     'rare',
     'news',
     'conference',
     'on',
     'Dec',
     'There',
     'was',
     'a',
     'discussion',
     'in',
     'the',
     'government',
     'about',
     'blocking',
     'social',
     'media',
     'sites',
     'and',
     'in',
     'the',
     'end',
     'it',
     'was',
     'decided',
     'to',
     'block',
     'them',
     'Users',
     'of',
     'the',
     'three',
     'main',
     'telecommunications',
     'operators',
     'in',
     'the',
     'country',
     'Zain',
     'MTN',
     'and',
     'Sudani',
     'said',
     'access',
     'to',
     'Facebook',
     'Twitter',
     'and',
     'WhatsApp',
     'has',
     'only',
     'been',
     'possible',
     'through',
     'use',
     'of',
     'a',
     'virtual',
     'private',
     'network',
     'VPN',
     'Though',
     'VPNs',
     'can',
     'bring',
     'their',
     'own',
     'connection',
     'problems',
     'and',
     'some',
     'Sudanese',
     'are',
     'unaware',
     'of',
     'their',
     'existence',
     'activists',
     'have',
     'used',
     'them',
     'widely',
     'to',
     'organise',
     'and',
     'document',
     'the',
     'demonstrations',
     'Hashtags',
     'in',
     'Arabic',
     'such',
     'as',
     'Sudan’s',
     'cities',
     'revolt',
     'have',
     'been',
     'widely',
     'circulated',
     'from',
     'Sudan',
     'and',
     'abroad',
     'Hashtags',
     'in',
     'English',
     'such',
     'as',
     'SudanRevolts',
     'have',
     'also',
     'been',
     'used',
     'Social',
     'media',
     'has',
     'a',
     'really',
     'big',
     'impact',
     'and',
     'it',
     'helps',
     'with',
     'forming',
     'public',
     'opinion',
     'and',
     'transmitting',
     'what’s',
     'happening',
     'in',
     'Sudan',
     'to',
     'the',
     'outside',
     'said',
     'Mujtaba',
     'Musa',
     'a',
     'Sudanese',
     'Twitter',
     'user',
     'with',
     'over',
     'followers',
     'who',
     'has',
     'been',
     'active',
     'in',
     'documenting',
     'the',
     'protests',
     'NetBlocks',
     'a',
     'digital',
     'rights',
     'NGO',
     'said',
     'data',
     'it',
     'collected',
     'including',
     'from',
     'thousands',
     'of',
     'Sudanese',
     'volunteers',
     'provided',
     'evidence',
     'of',
     'an',
     'extensive',
     'internet',
     'censorship',
     'regime',
     'Bader',
     'al',
     'Kharafi',
     'CEO',
     'of',
     'parent',
     'company',
     'Zain',
     'Group',
     'told',
     'Reuters',
     'Some',
     'websites',
     'may',
     'be',
     'blocked',
     'for',
     'technical',
     'reasons',
     'beyond',
     'the',
     'company’s',
     'specialisation',
     'Neither',
     'the',
     'National',
     'Telecommunications',
     'Corporation',
     'which',
     'oversees',
     'the',
     'sector',
     'in',
     'Sudan',
     'nor',
     'MTN',
     'or',
     'Sudani',
     'could',
     'be',
     'reached',
     'for',
     'comment',
     'Twitter',
     'and',
     'Facebook',
     'which',
     'also',
     'owns',
     'WhatsApp',
     'declined',
     'to',
     'comment',
     'While',
     'Sudan',
     'has',
     'a',
     'long',
     'history',
     'of',
     'systematically',
     'censoring',
     'print',
     'and',
     'broadcast',
     'media',
     'online',
     'media',
     'has',
     'been',
     'relatively',
     'untouched',
     'despite',
     'its',
     'exponential',
     'growth',
     'in',
     'recent',
     'years',
     'said',
     'Mai',
     'Truong',
     'of',
     'U',
     'S',
     'based',
     'advocacy',
     'group',
     'Freedom',
     'House',
     'The',
     'authorities',
     'have',
     'only',
     'now',
     'started',
     'to',
     'follow',
     'the',
     'playbook',
     'of',
     'other',
     'authoritarian',
     'governments',
     'Additional',
     'reporting',
     'by',
     'Ahmed',
     'Hagagy',
     'in',
     'Kuwait',
     'Editing',
     'by',
     'Aidan',
     'Lewis',
     'and',
     'Gareth',
     'Jones',
     'All',
     'quotes',
     'delayed',
     'a',
     'minimum',
     'of',
     'minutes',
     'See',
     'here',
     'for',
     'a',
     'complete',
     'list',
     'of',
     'exchanges',
     'and',
     'delays',
     'Reuters',
     'All',
     'Rights',
     'Reserved']




```python
print(f'{new_sample[8]} becomes: {lemmatizer.lemmatize(new_sample[8])}')
```

    media becomes: medium



```python
# However, look at the output below:
    
sentence = "He saw the trees get sawed down"
lemmed_sentence = [lemmatizer.lemmatize(token) for token in sentence.split(' ')]
lemmed_sentence
```




    ['He', 'saw', 'the', 'tree', 'get', 'sawed', 'down']




```python
one_random_student(student_first_names)
# What should have changed form but didn't?
```

    Ozair


Lemmatizers depend on POS tagging, and defaults to noun.

With a little bit of work, we can POS tag our text.


```python
pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
tokenizer = RegexpTokenizer(r"([a-zA-Z]+(?:[’'][a-z]+)?)")
new_sample = [token for token in new_sample if token not in custom_sw]
new_sample
```




    ['Sudanese',
     'authorities',
     'blocking',
     'access',
     'popular',
     'social',
     'media',
     'platforms',
     'used',
     'organise',
     'broadcast',
     'nationwide',
     'anti',
     'government',
     'protests',
     'triggered',
     'economic',
     'crisis',
     'internet',
     'users',
     'Sudan',
     'rocked',
     'near',
     'daily',
     'demonstrations',
     'past',
     'two',
     'weeks',
     'Protesters',
     'set',
     'alight',
     'ruling',
     'party',
     'buildings',
     'called',
     'President',
     'Omar',
     'al',
     'Bashir',
     'took',
     'power',
     'step',
     'In',
     'country',
     'state',
     'tightly',
     'controls',
     'traditional',
     'media',
     'internet',
     'become',
     'key',
     'information',
     'battleground',
     'Of',
     'Sudan’s',
     'million',
     'people',
     'million',
     'use',
     'internet',
     'million',
     'mobile',
     'phones',
     'local',
     'media',
     'Authorities',
     'repeated',
     'internet',
     'blackout',
     'imposed',
     'deadly',
     'protests',
     'But',
     'head',
     'Sudan’s',
     'National',
     'Intelligence',
     'Security',
     'Service',
     'Salah',
     'Abdallah',
     'told',
     'rare',
     'news',
     'conference',
     'Dec',
     'There',
     'discussion',
     'government',
     'blocking',
     'social',
     'media',
     'sites',
     'end',
     'decided',
     'block',
     'Users',
     'three',
     'main',
     'telecommunications',
     'operators',
     'country',
     'Zain',
     'MTN',
     'Sudani',
     'said',
     'access',
     'Facebook',
     'Twitter',
     'WhatsApp',
     'possible',
     'use',
     'virtual',
     'private',
     'network',
     'VPN',
     'Though',
     'VPNs',
     'bring',
     'connection',
     'problems',
     'Sudanese',
     'unaware',
     'existence',
     'activists',
     'used',
     'widely',
     'organise',
     'document',
     'demonstrations',
     'Hashtags',
     'Arabic',
     'Sudan’s',
     'cities',
     'revolt',
     'widely',
     'circulated',
     'Sudan',
     'abroad',
     'Hashtags',
     'English',
     'SudanRevolts',
     'also',
     'used',
     'Social',
     'media',
     'really',
     'big',
     'impact',
     'helps',
     'forming',
     'public',
     'opinion',
     'transmitting',
     'what’s',
     'happening',
     'Sudan',
     'outside',
     'said',
     'Mujtaba',
     'Musa',
     'Sudanese',
     'Twitter',
     'user',
     'followers',
     'active',
     'documenting',
     'protests',
     'NetBlocks',
     'digital',
     'rights',
     'NGO',
     'said',
     'data',
     'collected',
     'including',
     'thousands',
     'Sudanese',
     'volunteers',
     'provided',
     'evidence',
     'extensive',
     'internet',
     'censorship',
     'regime',
     'Bader',
     'al',
     'Kharafi',
     'CEO',
     'parent',
     'company',
     'Zain',
     'Group',
     'told',
     'Reuters',
     'Some',
     'websites',
     'may',
     'blocked',
     'technical',
     'reasons',
     'beyond',
     'company’s',
     'specialisation',
     'Neither',
     'National',
     'Telecommunications',
     'Corporation',
     'oversees',
     'sector',
     'Sudan',
     'MTN',
     'Sudani',
     'reached',
     'comment',
     'Twitter',
     'Facebook',
     'also',
     'owns',
     'WhatsApp',
     'declined',
     'comment',
     'While',
     'Sudan',
     'long',
     'history',
     'systematically',
     'censoring',
     'print',
     'broadcast',
     'media',
     'online',
     'media',
     'relatively',
     'untouched',
     'despite',
     'exponential',
     'growth',
     'recent',
     'years',
     'said',
     'Mai',
     'Truong',
     'U',
     'S',
     'based',
     'advocacy',
     'group',
     'Freedom',
     'House',
     'The',
     'authorities',
     'started',
     'follow',
     'playbook',
     'authoritarian',
     'governments',
     'Additional',
     'reporting',
     'Ahmed',
     'Hagagy',
     'Kuwait',
     'Editing',
     'Aidan',
     'Lewis',
     'Gareth',
     'Jones',
     'All',
     'quotes',
     'delayed',
     'minimum',
     'minutes',
     'See',
     'complete',
     'list',
     'exchanges',
     'delays',
     'Reuters',
     'All',
     'Rights',
     'Reserved']




```python
nltk.download('tagsets')
nltk.help.upenn_tagset()
```

    $: dollar
        $ -$ --$ A$ C$ HK$ M$ NZ$ S$ U.S.$ US$
    '': closing quotation mark
        ' ''
    (: opening parenthesis
        ( [ {
    ): closing parenthesis
        ) ] }
    ,: comma
        ,
    --: dash
        --
    .: sentence terminator
        . ! ?
    :: colon or ellipsis
        : ; ...
    CC: conjunction, coordinating
        & 'n and both but either et for less minus neither nor or plus so
        therefore times v. versus vs. whether yet
    CD: numeral, cardinal
        mid-1890 nine-thirty forty-two one-tenth ten million 0.5 one forty-
        seven 1987 twenty '79 zero two 78-degrees eighty-four IX '60s .025
        fifteen 271,124 dozen quintillion DM2,000 ...
    DT: determiner
        all an another any both del each either every half la many much nary
        neither no some such that the them these this those
    EX: existential there
        there
    FW: foreign word
        gemeinschaft hund ich jeux habeas Haementeria Herr K'ang-si vous
        lutihaw alai je jour objets salutaris fille quibusdam pas trop Monte
        terram fiche oui corporis ...
    IN: preposition or conjunction, subordinating
        astride among uppon whether out inside pro despite on by throughout
        below within for towards near behind atop around if like until below
        next into if beside ...
    JJ: adjective or numeral, ordinal
        third ill-mannered pre-war regrettable oiled calamitous first separable
        ectoplasmic battery-powered participatory fourth still-to-be-named
        multilingual multi-disciplinary ...
    JJR: adjective, comparative
        bleaker braver breezier briefer brighter brisker broader bumper busier
        calmer cheaper choosier cleaner clearer closer colder commoner costlier
        cozier creamier crunchier cuter ...
    JJS: adjective, superlative
        calmest cheapest choicest classiest cleanest clearest closest commonest
        corniest costliest crassest creepiest crudest cutest darkest deadliest
        dearest deepest densest dinkiest ...
    LS: list item marker
        A A. B B. C C. D E F First G H I J K One SP-44001 SP-44002 SP-44005
        SP-44007 Second Third Three Two * a b c d first five four one six three
        two
    MD: modal auxiliary
        can cannot could couldn't dare may might must need ought shall should
        shouldn't will would
    NN: noun, common, singular or mass
        common-carrier cabbage knuckle-duster Casino afghan shed thermostat
        investment slide humour falloff slick wind hyena override subhumanity
        machinist ...
    NNP: noun, proper, singular
        Motown Venneboerger Czestochwa Ranzer Conchita Trumplane Christos
        Oceanside Escobar Kreisler Sawyer Cougar Yvette Ervin ODI Darryl CTCA
        Shannon A.K.C. Meltex Liverpool ...
    NNPS: noun, proper, plural
        Americans Americas Amharas Amityvilles Amusements Anarcho-Syndicalists
        Andalusians Andes Andruses Angels Animals Anthony Antilles Antiques
        Apache Apaches Apocrypha ...
    NNS: noun, common, plural
        undergraduates scotches bric-a-brac products bodyguards facets coasts
        divestitures storehouses designs clubs fragrances averages
        subjectivists apprehensions muses factory-jobs ...
    PDT: pre-determiner
        all both half many quite such sure this
    POS: genitive marker
        ' 's
    PRP: pronoun, personal
        hers herself him himself hisself it itself me myself one oneself ours
        ourselves ownself self she thee theirs them themselves they thou thy us
    PRP$: pronoun, possessive
        her his mine my our ours their thy your
    RB: adverb
        occasionally unabatingly maddeningly adventurously professedly
        stirringly prominently technologically magisterially predominately
        swiftly fiscally pitilessly ...
    RBR: adverb, comparative
        further gloomier grander graver greater grimmer harder harsher
        healthier heavier higher however larger later leaner lengthier less-
        perfectly lesser lonelier longer louder lower more ...
    RBS: adverb, superlative
        best biggest bluntest earliest farthest first furthest hardest
        heartiest highest largest least less most nearest second tightest worst
    RP: particle
        aboard about across along apart around aside at away back before behind
        by crop down ever fast for forth from go high i.e. in into just later
        low more off on open out over per pie raising start teeth that through
        under unto up up-pp upon whole with you
    SYM: symbol
        % & ' '' ''. ) ). * + ,. < = > @ A[fj] U.S U.S.S.R * ** ***
    TO: "to" as preposition or infinitive marker
        to
    UH: interjection
        Goodbye Goody Gosh Wow Jeepers Jee-sus Hubba Hey Kee-reist Oops amen
        huh howdy uh dammit whammo shucks heck anyways whodunnit honey golly
        man baby diddle hush sonuvabitch ...
    VB: verb, base form
        ask assemble assess assign assume atone attention avoid bake balkanize
        bank begin behold believe bend benefit bevel beware bless boil bomb
        boost brace break bring broil brush build ...
    VBD: verb, past tense
        dipped pleaded swiped regummed soaked tidied convened halted registered
        cushioned exacted snubbed strode aimed adopted belied figgered
        speculated wore appreciated contemplated ...
    VBG: verb, present participle or gerund
        telegraphing stirring focusing angering judging stalling lactating
        hankerin' alleging veering capping approaching traveling besieging
        encrypting interrupting erasing wincing ...
    VBN: verb, past participle
        multihulled dilapidated aerosolized chaired languished panelized used
        experimented flourished imitated reunifed factored condensed sheared
        unsettled primed dubbed desired ...
    VBP: verb, present tense, not 3rd person singular
        predominate wrap resort sue twist spill cure lengthen brush terminate
        appear tend stray glisten obtain comprise detest tease attract
        emphasize mold postpone sever return wag ...
    VBZ: verb, present tense, 3rd person singular
        bases reconstructs marks mixes displeases seals carps weaves snatches
        slumps stretches authorizes smolders pictures emerges stockpiles
        seduces fizzes uses bolsters slaps speaks pleads ...
    WDT: WH-determiner
        that what whatever which whichever
    WP: WH-pronoun
        that what whatever whatsoever which who whom whosoever
    WP$: WH-pronoun, possessive
        whose
    WRB: Wh-adverb
        how however whence whenever where whereby whereever wherein whereof why
    ``: opening quotation mark
        ` ``


    [nltk_data] Downloading package tagsets to
    [nltk_data]     /Users/johnmaxbarry/nltk_data...
    [nltk_data]   Package tagsets is already up-to-date!



```python
from nltk import pos_tag
# Use nltk's pos_tag to tag our words
# Does a pretty good job, but does make some mistakes

```


```python
#__SOLUTION__
new_sample_tagged = pos_tag(new_sample)
new_sample_tagged
```




    [('Sudanese', 'JJ'),
     ('authorities', 'NNS'),
     ('blocking', 'VBG'),
     ('access', 'NN'),
     ('popular', 'JJ'),
     ('social', 'JJ'),
     ('media', 'NNS'),
     ('platforms', 'NNS'),
     ('used', 'VBD'),
     ('organise', 'RB'),
     ('broadcast', 'JJ'),
     ('nationwide', 'JJ'),
     ('anti', 'JJ'),
     ('government', 'NN'),
     ('protests', 'NNS'),
     ('triggered', 'VBD'),
     ('economic', 'JJ'),
     ('crisis', 'NN'),
     ('internet', 'NN'),
     ('users', 'NNS'),
     ('Sudan', 'NNP'),
     ('rocked', 'VBD'),
     ('near', 'IN'),
     ('daily', 'JJ'),
     ('demonstrations', 'NNS'),
     ('past', 'IN'),
     ('two', 'CD'),
     ('weeks', 'NNS'),
     ('Protesters', 'NNPS'),
     ('set', 'VBD'),
     ('alight', 'RP'),
     ('ruling', 'VBG'),
     ('party', 'NN'),
     ('buildings', 'NNS'),
     ('called', 'VBN'),
     ('President', 'NNP'),
     ('Omar', 'NNP'),
     ('al', 'NN'),
     ('Bashir', 'NNP'),
     ('took', 'VBD'),
     ('power', 'NN'),
     ('step', 'NN'),
     ('In', 'IN'),
     ('country', 'NN'),
     ('state', 'NN'),
     ('tightly', 'RB'),
     ('controls', 'VBZ'),
     ('traditional', 'JJ'),
     ('media', 'NNS'),
     ('internet', 'NN'),
     ('become', 'VBP'),
     ('key', 'JJ'),
     ('information', 'NN'),
     ('battleground', 'NN'),
     ('Of', 'IN'),
     ('Sudan’s', 'NNP'),
     ('million', 'CD'),
     ('people', 'NNS'),
     ('million', 'CD'),
     ('use', 'JJ'),
     ('internet', 'NN'),
     ('million', 'CD'),
     ('mobile', 'JJ'),
     ('phones', 'NNS'),
     ('local', 'JJ'),
     ('media', 'NNS'),
     ('Authorities', 'NNP'),
     ('repeated', 'VBD'),
     ('internet', 'NN'),
     ('blackout', 'NN'),
     ('imposed', 'VBN'),
     ('deadly', 'JJ'),
     ('protests', 'NNS'),
     ('But', 'CC'),
     ('head', 'VBP'),
     ('Sudan’s', 'NNP'),
     ('National', 'NNP'),
     ('Intelligence', 'NNP'),
     ('Security', 'NNP'),
     ('Service', 'NNP'),
     ('Salah', 'NNP'),
     ('Abdallah', 'NNP'),
     ('told', 'VBD'),
     ('rare', 'JJ'),
     ('news', 'NN'),
     ('conference', 'NN'),
     ('Dec', 'NNP'),
     ('There', 'EX'),
     ('discussion', 'JJ'),
     ('government', 'NN'),
     ('blocking', 'VBG'),
     ('social', 'JJ'),
     ('media', 'NNS'),
     ('sites', 'NNS'),
     ('end', 'VBP'),
     ('decided', 'VBD'),
     ('block', 'NN'),
     ('Users', 'NNS'),
     ('three', 'CD'),
     ('main', 'JJ'),
     ('telecommunications', 'NN'),
     ('operators', 'NNS'),
     ('country', 'NN'),
     ('Zain', 'NNP'),
     ('MTN', 'NNP'),
     ('Sudani', 'NNP'),
     ('said', 'VBD'),
     ('access', 'NN'),
     ('Facebook', 'NNP'),
     ('Twitter', 'NNP'),
     ('WhatsApp', 'NNP'),
     ('possible', 'JJ'),
     ('use', 'NN'),
     ('virtual', 'JJ'),
     ('private', 'JJ'),
     ('network', 'NN'),
     ('VPN', 'NNP'),
     ('Though', 'NNP'),
     ('VPNs', 'NNP'),
     ('bring', 'VBG'),
     ('connection', 'NN'),
     ('problems', 'NNS'),
     ('Sudanese', 'JJ'),
     ('unaware', 'JJ'),
     ('existence', 'NN'),
     ('activists', 'NNS'),
     ('used', 'VBD'),
     ('widely', 'RB'),
     ('organise', 'JJ'),
     ('document', 'NN'),
     ('demonstrations', 'NNS'),
     ('Hashtags', 'NNP'),
     ('Arabic', 'NNP'),
     ('Sudan’s', 'NNP'),
     ('cities', 'NNS'),
     ('revolt', 'VBP'),
     ('widely', 'RB'),
     ('circulated', 'VBN'),
     ('Sudan', 'NNP'),
     ('abroad', 'RB'),
     ('Hashtags', 'NNP'),
     ('English', 'NNP'),
     ('SudanRevolts', 'NNP'),
     ('also', 'RB'),
     ('used', 'VBD'),
     ('Social', 'NNP'),
     ('media', 'NNS'),
     ('really', 'RB'),
     ('big', 'JJ'),
     ('impact', 'NN'),
     ('helps', 'VBZ'),
     ('forming', 'VBG'),
     ('public', 'JJ'),
     ('opinion', 'NN'),
     ('transmitting', 'VBG'),
     ('what’s', 'JJ'),
     ('happening', 'VBG'),
     ('Sudan', 'NNP'),
     ('outside', 'NN'),
     ('said', 'VBD'),
     ('Mujtaba', 'NNP'),
     ('Musa', 'NNP'),
     ('Sudanese', 'NNP'),
     ('Twitter', 'NNP'),
     ('user', 'NN'),
     ('followers', 'NNS'),
     ('active', 'JJ'),
     ('documenting', 'VBG'),
     ('protests', 'NNS'),
     ('NetBlocks', 'NNP'),
     ('digital', 'JJ'),
     ('rights', 'NNS'),
     ('NGO', 'NNP'),
     ('said', 'VBD'),
     ('data', 'NNS'),
     ('collected', 'VBD'),
     ('including', 'VBG'),
     ('thousands', 'NNS'),
     ('Sudanese', 'JJ'),
     ('volunteers', 'NNS'),
     ('provided', 'VBD'),
     ('evidence', 'NN'),
     ('extensive', 'JJ'),
     ('internet', 'NN'),
     ('censorship', 'NN'),
     ('regime', 'NN'),
     ('Bader', 'NNP'),
     ('al', 'NN'),
     ('Kharafi', 'NNP'),
     ('CEO', 'NNP'),
     ('parent', 'NN'),
     ('company', 'NN'),
     ('Zain', 'NNP'),
     ('Group', 'NNP'),
     ('told', 'VBD'),
     ('Reuters', 'NNP'),
     ('Some', 'DT'),
     ('websites', 'NNS'),
     ('may', 'MD'),
     ('blocked', 'VB'),
     ('technical', 'JJ'),
     ('reasons', 'NNS'),
     ('beyond', 'IN'),
     ('company’s', 'JJ'),
     ('specialisation', 'NN'),
     ('Neither', 'NNP'),
     ('National', 'NNP'),
     ('Telecommunications', 'NNP'),
     ('Corporation', 'NNP'),
     ('oversees', 'VBZ'),
     ('sector', 'NN'),
     ('Sudan', 'NNP'),
     ('MTN', 'NNP'),
     ('Sudani', 'NNP'),
     ('reached', 'VBD'),
     ('comment', 'NN'),
     ('Twitter', 'NNP'),
     ('Facebook', 'NNP'),
     ('also', 'RB'),
     ('owns', 'VBZ'),
     ('WhatsApp', 'NNP'),
     ('declined', 'VBD'),
     ('comment', 'NN'),
     ('While', 'IN'),
     ('Sudan', 'NNP'),
     ('long', 'JJ'),
     ('history', 'NN'),
     ('systematically', 'RB'),
     ('censoring', 'VBG'),
     ('print', 'NN'),
     ('broadcast', 'NN'),
     ('media', 'NNS'),
     ('online', 'JJ'),
     ('media', 'NNS'),
     ('relatively', 'RB'),
     ('untouched', 'JJ'),
     ('despite', 'IN'),
     ('exponential', 'JJ'),
     ('growth', 'NN'),
     ('recent', 'JJ'),
     ('years', 'NNS'),
     ('said', 'VBD'),
     ('Mai', 'NNP'),
     ('Truong', 'NNP'),
     ('U', 'NNP'),
     ('S', 'NNP'),
     ('based', 'VBN'),
     ('advocacy', 'NN'),
     ('group', 'NN'),
     ('Freedom', 'NNP'),
     ('House', 'NNP'),
     ('The', 'DT'),
     ('authorities', 'NNS'),
     ('started', 'VBD'),
     ('follow', 'JJ'),
     ('playbook', 'NN'),
     ('authoritarian', 'JJ'),
     ('governments', 'NNS'),
     ('Additional', 'JJ'),
     ('reporting', 'NN'),
     ('Ahmed', 'NNP'),
     ('Hagagy', 'NNP'),
     ('Kuwait', 'NNP'),
     ('Editing', 'NNP'),
     ('Aidan', 'NNP'),
     ('Lewis', 'NNP'),
     ('Gareth', 'NNP'),
     ('Jones', 'NNP'),
     ('All', 'NNP'),
     ('quotes', 'VBZ'),
     ('delayed', 'VBN'),
     ('minimum', 'JJ'),
     ('minutes', 'NNS'),
     ('See', 'VBP'),
     ('complete', 'JJ'),
     ('list', 'NN'),
     ('exchanges', 'NNS'),
     ('delays', 'VBP'),
     ('Reuters', 'NNP'),
     ('All', 'NNP'),
     ('Rights', 'NNP'),
     ('Reserved', 'VBD')]




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
#__SOLUTION__
new_sample_lemmed = [lemmatizer.lemmatize(token[0], token[1]) for token in new_sample_tagged]
```


```python
new_sample_lemmed[:10]
```




    ['Sudanese',
     'authority',
     'block',
     'access',
     'popular',
     'social',
     'medium',
     'platform',
     'use',
     'organise']




```python
print(f'There are {len(set(new_sample_lemmed))} unique lemmas')
```

    There are 162 unique lemmas



```python
fdist = FreqDist(new_sample_lemmed)
plt.figure(figsize=(10,10))
fdist.plot(30)
```


![png](index_files/index_147_0.png)





    <matplotlib.axes._subplots.AxesSubplot at 0x1a2fa2e828>



## 4. Feature Engineering for NLP 
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

# instantiate a count vectorizer
vec = None

# fit vectorizor on our lemmed sample. Note, the vectorizer takes in raw texts, so we need to join all of our lemmed tokens.
X = 
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
      <th>account</th>
      <th>add</th>
      <th>additional</th>
      <th>africa</th>
      <th>almost</th>
      <th>already</th>
      <th>also</th>
      <th>although</th>
      <th>amid</th>
      <th>around</th>
      <th>...</th>
      <th>union</th>
      <th>vessel</th>
      <th>vote</th>
      <th>want</th>
      <th>week</th>
      <th>well</th>
      <th>work</th>
      <th>world</th>
      <th>would</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 159 columns</p>
</div>




```python
#__SOLUTION__
vec = CountVectorizer()
X = vec.fit_transform([" ".join(new_sample_lemmed)])


df = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
df.head()
```

That is not very exciting for one document. The idea is to make a document term matrix for all of the words in our corpus.


```python
corpus
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
      <th>body</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Noting that the resignation of James Mattis as...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Desperate to unwind after months of nonstop wo...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nearly halfway through his presidential term, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Attempting to make amends for gross abuses of ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Decrying the Senate’s resolution blaming the c...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>Britain’s opposition leader Jeremy Corbyn wou...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>996</th>
      <td>Turkey will take over the fight against Islam...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>997</th>
      <td>Malaysia is seeking $7.5 billion in reparatio...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>998</th>
      <td>An Israeli court sentenced a Palestinian to 1...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>999</th>
      <td>At least 22 people have died due to landslide...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 2 columns</p>
</div>



We can pass in arguments such as a regex pattern, a list of stopwords, and an ngram range to do our preprocessing in one fell swoop.   
*Note lowercase defaults to true.*


```python
# pass in the regex from above, our cusomt stopwords, and an ngram range: [1,1], [1,2] , [1,3]
```


```python
#__SOLUTION__
vec = CountVectorizer(token_pattern=r"([a-zA-Z]+(?:'[a-z]+)?)", stop_words=custom_sw, ngram_range=[1,2])
X = vec.fit_transform(corpus.body[0:2])

df = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
df.head()
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
      <th>adding</th>
      <th>adding wants</th>
      <th>administration</th>
      <th>administration official</th>
      <th>advisor</th>
      <th>advisor michael</th>
      <th>also</th>
      <th>also noting</th>
      <th>americans</th>
      <th>americans also</th>
      <th>...</th>
      <th>witnesses want</th>
      <th>work</th>
      <th>work investigating</th>
      <th>worried</th>
      <th>worried populace</th>
      <th>year</th>
      <th>year country</th>
      <th>year old</th>
      <th>yet</th>
      <th>yet another</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 371 columns</p>
</div>



Our document term matrix gets bigger and bigger, with more and more zeros, becoming sparser and sparser.

> In case you forgot, a sparse matrix "is a matrix in which most of the elements are zero." [wikipedia](https://en.wikipedia.org/wiki/Sparse_matrix)

We can set upper and lower limits to the word frequency.


```python
vec = CountVectorizer(token_pattern=r"([a-zA-Z]+(?:'[a-z]+)?)", stop_words=custom_sw, ngram_range=[1,2], min_df=2, max_df=25)
X = vec.fit_transform(corpus.body)

df_cv = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
df_cv
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
      <th>aapl</th>
      <th>aaron</th>
      <th>aaron ross</th>
      <th>ab</th>
      <th>abandon</th>
      <th>abandon conservatives</th>
      <th>abandoned</th>
      <th>abandoned grassroots</th>
      <th>abandoning</th>
      <th>abandoning quarter</th>
      <th>...</th>
      <th>zone</th>
      <th>zone eu</th>
      <th>zones</th>
      <th>zoo</th>
      <th>zoo closed</th>
      <th>zooming</th>
      <th>zor</th>
      <th>zte</th>
      <th>zte corp</th>
      <th>zuckerberg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>996</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>997</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>998</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>999</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 31043 columns</p>
</div>



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
      <th>aa</th>
      <th>aaaaaaah</th>
      <th>aaaaaah</th>
      <th>aaaaargh</th>
      <th>aaaah</th>
      <th>aaah</th>
      <th>aaargh</th>
      <th>aah</th>
      <th>aahing</th>
      <th>aap</th>
      <th>...</th>
      <th>zoos</th>
      <th>zor</th>
      <th>zozovitch</th>
      <th>zte</th>
      <th>zuckerberg</th>
      <th>zuercher</th>
      <th>zverev</th>
      <th>zych</th>
      <th>zzouss</th>
      <th>zzzzzst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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
<p>5 rows × 23455 columns</p>
</div>




```python
#__SOLUTION__
tf_vec = TfidfVectorizer(token_pattern=r"([a-zA-Z]+(?:'[a-z]+)?)", stop_words=custom_sw)
X = tf_vec.fit_transform(corpus.body)

df = pd.DataFrame(X.toarray(), columns = tf_vec.get_feature_names())
df.head()
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
      <th>aa</th>
      <th>aaaaaaah</th>
      <th>aaaaaah</th>
      <th>aaaaargh</th>
      <th>aaaah</th>
      <th>aaah</th>
      <th>aaargh</th>
      <th>aah</th>
      <th>aahing</th>
      <th>aap</th>
      <th>...</th>
      <th>zoos</th>
      <th>zor</th>
      <th>zozovitch</th>
      <th>zte</th>
      <th>zuckerberg</th>
      <th>zuercher</th>
      <th>zverev</th>
      <th>zych</th>
      <th>zzouss</th>
      <th>zzzzzst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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
<p>5 rows × 23455 columns</p>
</div>




```python
corpus.iloc[313].body
```




    'Power nerds, these are a special breed of techie nerds who want to take over the world with their technological prowess. From the birth of the World Wide Web, these power nerds have created and grown companies that are now monopolies controlling every facet of people’s lives. Smartphones, apps, search engines, social networks – power nerds are in everything, their power increases daily as more millions of people use their networks. Power nerds are ruthless, they are creatures who do not balk in crushing their opponents completely without mercy and their greed for complete controlling power over everything is boundless. To quote a few examples of companies that are run by ruthless power nerds, we can of course cite Facebook, Twitter, Amazon and Google. These companies are not only seeking to rule and control everything, they also are using their power to manipulate data taken from their platforms to make money and increase their influence, as well as shut down any voices that are not left leaning. All Hail Zuckerberg “Power nerds are inherently evil. Zuckerberg is one example of a power nerd so power hungry that he pursues global domination with a vehement ruthless nasty streak. These tech robots are machines, they are not really human anymore, their fuel is pure power and more power, and they will use billions of people to achieve their goals at all cost. Tech power nerds are farmers of people, they farm billions of people for data,” an observer of the current situation revealed. Power nerds also do not have a problem about farming data from billions of people without their knowledge, they also abuse their positions to be politically biased and censor free speech as a means of gaining even more power. One can only hope that companies like Facebook one day are brought to justice for their evil, devious crimes committed against billions of people. In 2009, Facebook was caught lying to their account holders about the amount and type of information it was collecting on them, and the company also explicitly lied about who they were providing that information to. As a result, the Federal Trade Commission censured the company in 2011 for violations of Article 5 of the Federal Trade Commission Act. The core mission of Article 5 of the FTC Act is to protect consumer welfare and prevent unfair business acts or practices from occurring. In defiance of the Federal Trade Commission’s order, Facebook continued to reveal their customers private account information to unauthorized individuals and corporations and is therefore liable for civil penalties of $41,484 per each violation – a fine that could reach $3 trillion Dollars. Talking about fines, Google, a company that controls 91.5% of search traffic in Europe alone, is being slapped with antitrust fines from the EU, but it’s only for a measly £2.14 billion, which for a company that pays literally no tax, is peanuts. At the end of the day, these companies led by power nerds have now spread their octopus-like grip over the whole globe, and to even begin deconstructing their evil plan of complete control, will be nearly impossible now unless these companies are fined, broken up and told to pay the tax they owe. Hopefully one day the power nerds are put in their place, and we can all breathe a breath of fresh air on a free internet once again.'




```python
df.iloc[313].sort_values(ascending=False)[:10]
```




    nerds         0.601117
    power         0.396907
    companies     0.168315
    billions      0.140886
    facebook      0.129390
    ruthless      0.127374
    company       0.106838
    evil          0.104807
    zuckerberg    0.098795
    people        0.095754
    Name: 313, dtype: float64



Let's compare the tfidf to the count vectorizer output for one document.


```python
vec = CountVectorizer(token_pattern=r"([a-zA-Z]+(?:'[a-z]+)?)", stop_words=custom_sw)
X = vec.fit_transform(corpus.body)

df_cv = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
df_cv
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
      <th>aa</th>
      <th>aaaaaaah</th>
      <th>aaaaaah</th>
      <th>aaaaargh</th>
      <th>aaaah</th>
      <th>aaah</th>
      <th>aaargh</th>
      <th>aah</th>
      <th>aahing</th>
      <th>aap</th>
      <th>...</th>
      <th>zoos</th>
      <th>zor</th>
      <th>zozovitch</th>
      <th>zte</th>
      <th>zuckerberg</th>
      <th>zuercher</th>
      <th>zverev</th>
      <th>zych</th>
      <th>zzouss</th>
      <th>zzzzzst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>996</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>997</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>998</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>999</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 23455 columns</p>
</div>




```python
df_cv.iloc[313].sort_values(ascending=False)[:10]
```




    power        18
    nerds        11
    people        7
    companies     6
    billions      4
    facebook      4
    also          4
    company       4
    trade         3
    day           3
    Name: 313, dtype: int64



The tfidf lessoned the importance of some of the more common words, including a stopword which "also" which didn't make it into the stopword list.

It also assigns "nerds" more weight than power.  


```python
print(f'Nerds only shows up in document 313: {len(df_cv[df.nerds!=0])}')
print(f'Power shows up in {len(df_cv[df.power!=0])}')
```

    Nerds only shows up in document 313: 1
    Power shows up in 147



```python
tf_vec.vocabulary_
```




    {'noting': 14227,
     'resignation': 17451,
     'james': 11082,
     'mattis': 12875,
     'secretary': 18490,
     'defense': 5327,
     'marked': 12752,
     'ouster': 14709,
     'third': 21059,
     'top': 21288,
     'administration': 293,
     'official': 14444,
     'less': 12041,
     'three': 21093,
     'weeks': 22856,
     'worried': 23183,
     'populace': 15782,
     'told': 21254,
     'reporters': 17363,
     'friday': 8388,
     'unsure': 22123,
     'many': 12704,
     'former': 8260,
     'trump': 21630,
     'staffers': 19829,
     'safely': 18070,
     'reabsorb': 16836,
     'jesus': 11158,
     'take': 20690,
     'back': 1540,
     'assholes': 1254,
     'need': 13935,
     'time': 21174,
     'process': 16144,
     'get': 8761,
     'next': 14044,
     'said': 18086,
     'year': 23298,
     'old': 14476,
     'gregory': 9105,
     'birch': 2136,
     'naperville': 13829,
     'il': 10219,
     'echoing': 6554,
     'concerns': 4203,
     'million': 13239,
     'americans': 723,
     'also': 658,
     'country': 4673,
     'truly': 21629,
     'beginning': 1905,
     'reintegrate': 17180,
     'national': 13869,
     'security': 18508,
     'advisor': 365,
     'michael': 13157,
     'flynn': 8140,
     'sustainable': 20525,
     'handle': 9391,
     'maybe': 12899,
     'two': 21760,
     'members': 13033,
     'inner': 10654,
     'circle': 3622,
     'remainder': 17246,
     'limits': 12175,
     'u': 21780,
     'confirmed': 4262,
     'pieces': 15464,
     'shit': 18875,
     'trying': 21647,
     'rejoin': 17195,
     'society': 19400,
     'desperate': 5612,
     'unwind': 22151,
     'months': 13508,
     'nonstop': 14170,
     'work': 23162,
     'investigating': 10906,
     'russian': 18011,
     'influence': 10581,
     'election': 6660,
     'visibly': 22537,
     'exhausted': 7344,
     'special': 19600,
     'counsel': 4656,
     'robert': 17790,
     'mueller': 13660,
     'powered': 15891,
     'phone': 15413,
     'order': 14629,
     'give': 8823,
     'break': 2565,
     'news': 14035,
     'concerning': 4202,
     'probe': 16131,
     'holiday': 9844,
     'last': 11835,
     'thing': 21047,
     'want': 22722,
     'spending': 19649,
     'family': 7624,
     'cascade': 3151,
     'push': 16489,
     'notifications': 14224,
     'telling': 20884,
     'yet': 23325,
     'another': 885,
     'oligarch': 14487,
     'political': 15733,
     'operative': 14565,
     'highly': 9750,
     'placed': 15557,
     'socialite': 19396,
     'used': 22231,
     'deutsche': 5688,
     'bank': 1676,
     'channels': 3382,
     'funnel': 8495,
     'money': 13472,
     'campaign': 2971,
     'fbi': 7736,
     'director': 5870,
     'firmly': 7978,
     'holding': 9838,
     'power': 15890,
     'button': 2865,
     'adding': 256,
     'wants': 22725,
     'completely': 4131,
     'present': 16002,
     'moment': 13454,
     'celebrating': 3264,
     'loved': 12394,
     'ones': 14516,
     'ruminating': 17976,
     'met': 13110,
     'diplomat': 5855,
     'whether': 22935,
     'someone': 19462,
     'using': 22242,
     'social': 19390,
     'media': 12974,
     'tamper': 20727,
     'witnesses': 23114,
     'calm': 2949,
     'even': 7225,
     'think': 21051,
     'individual': 10506,
     'name': 13815,
     'wait': 22675,
     'hear': 9573,
     'important': 10340,
     'developments': 5703,
     'january': 11101,
     'since': 19083,
     'know': 11617,
     'second': 18481,
     'read': 16852,
     'something': 19465,
     'eric': 7091,
     'involved': 10932,
     'deeply': 5296,
     'previously': 16052,
     'suspected': 20511,
     'pulled': 16402,
     'ruin': 17959,
     'whole': 22975,
     'vacation': 22274,
     'press': 16018,
     'reactivated': 16848,
     'check': 3449,
     'real': 16869,
     'quick': 16603,
     'nearly': 13920,
     'halfway': 9340,
     'presidential': 16016,
     'term': 20938,
     'donald': 6216,
     'continued': 4456,
     'exist': 7355,
     'perpetual': 15307,
     'state': 19911,
     'controversy': 4495,
     'provided': 16334,
     'shortage': 18912,
     'outrageous': 14749,
     'moments': 13457,
     'onion': 14521,
     'looks': 12344,
     'significant': 19037,
     'events': 7230,
     'presidency': 16014,
     'attempting': 1341,
     'make': 12592,
     'amends': 719,
     'gross': 9151,
     'abuses': 101,
     'interior': 10819,
     'department': 5520,
     'unusually': 22141,
     'contrite': 4486,
     'ryan': 18022,
     'zinke': 23431,
     'apologized': 957,
     'monday': 13467,
     'misusing': 13357,
     'government': 8991,
     'funds': 8490,
     'sending': 18587,
     'ethics': 7172,
     'committee': 4045,
     'vase': 22333,
     'change': 3377,
     'anything': 931,
     'exploited': 7423,
     'cabinet': 2896,
     'position': 15830,
     'hope': 9922,
     'accept': 121,
     'beautiful': 1863,
     'example': 7279,
     'qing': 16541,
     'dynasty': 6491,
     'porcelain': 15796,
     'small': 19286,
     'token': 21251,
     'regret': 17134,
     'acknowledging': 198,
     'gift': 8792,
     'spent': 19651,
     'taxpayer': 20804,
     'renovate': 17306,
     'office': 14440,
     'doors': 6244,
     'hoped': 9923,
     'would': 23201,
     'consider': 4344,
     'sincere': 19084,
     'gesture': 8757,
     'apology': 958,
     'wrong': 23236,
     'advantage': 337,
     'lustrous': 12463,
     'glazing': 8843,
     'firing': 7974,
     'evident': 7249,
     'piece': 15462,
     'move': 13611,
     'forgive': 8242,
     'human': 10047,
     'failings': 7580,
     'please': 15619,
     'remember': 17261,
     'man': 12631,
     'detail': 5642,
     'turkey': 21701,
     'violated': 22505,
     'hatch': 9503,
     'act': 218,
     'acted': 219,
     'pawn': 15148,
     'oil': 14465,
     'gas': 8619,
     'industry': 10534,
     'rather': 16799,
     'eyes': 7520,
     'happen': 9419,
     'fall': 7605,
     'unique': 22009,
     'kaolin': 11394,
     'clay': 3717,
     'bought': 2470,
     'mercedes': 13070,
     'benz': 2000,
     'sedans': 18510,
     'find': 7938,
     'parking': 15025,
     'lot': 12375,
     'leave': 11942,
     'today': 21240,
     'plans': 15583,
     'apologize': 956,
     'person': 15322,
     'member': 13032,
     'visiting': 22542,
     'homes': 9870,
     'helicopter': 9641,
     'decrying': 5281,
     'senate': 18583,
     'resolution': 17463,
     'blaming': 2196,
     'crown': 4881,
     'prince': 16084,
     'brutal': 2707,
     'torture': 21320,
     'murder': 13713,
     'journalist': 11259,
     'jamal': 11081,
     'khashoggi': 11496,
     'cruel': 4891,
     'inhumane': 10627,
     'unprecedented': 22063,
     'interference': 10815,
     'sovereign': 19538,
     'kingdom': 11551,
     'internal': 10828,
     'affairs': 385,
     'launched': 11860,
     'rights': 17708,
     'investigation': 10907,
     'harsh': 9478,
     'treatment': 21522,
     'saudi': 18232,
     'ruler': 17968,
     'mohammad': 13436,
     'bin': 2114,
     'salman': 18120,
     'looking': 12342,
     'troubling': 21612,
     'accusations': 175,
     'united': 22016,
     'states': 19916,
     'chosen': 3574,
     'willfully': 23024,
     'knowingly': 11620,
     'place': 15556,
     'fault': 7716,
     'dissident': 6075,
     'president': 16015,
     'claiming': 3666,
     'despot': 5620,
     'made': 12516,
     'endure': 6886,
     'loss': 12372,
     'military': 13221,
     'funding': 8488,
     'ongoing': 14520,
     'war': 22727,
     'yemen': 23315,
     'left': 11960,
     'millions': 13241,
     'homeless': 9868,
     'starving': 19908,
     'matter': 12870,
     'whose': 22985,
     'dismemberment': 6009,
     'may': 12897,
     'ordered': 14630,
     'facing': 7550,
     'criticism': 4845,
     'like': 12152,
     'international': 10830,
     'stage': 19833,
     'powerful': 15892,
     'leader': 11910,
     'basically': 1775,
     'kind': 11542,
     'mistreatment': 13351,
     'seriously': 18659,
     'treat': 21518,
     'authoritarian': 1414,
     'regimes': 17120,
     'purchase': 16454,
     'weapons': 22822,
     'without': 23110,
     'billions': 2109,
     'dollars': 6200,
     'aid': 476,
     'regime': 17119,
     'supposed': 20448,
     'maintain': 12574,
     'basic': 1774,
     'standard': 19870,
     'living': 12261,
     'expected': 7378,
     'charge': 3406,
     'american': 722,
     'senators': 18585,
     'crimes': 4816,
     'humanity': 10051,
     'role': 17831,
     'responsible': 17498,
     'actions': 222,
     'following': 8173,
     'sentencing': 18616,
     'hush': 10104,
     'scandal': 18271,
     'cohen': 3891,
     'granted': 9045,
     'prison': 16106,
     'release': 17213,
     'new': 14025,
     'job': 11201,
     'sources': 19521,
     'wednesday': 22848,
     'confident': 4254,
     'engaging': 6903,
     'honest': 9889,
     'help': 9653,
     'mr': 13631,
     'rehabilitation': 17153,
     'warden': 22730,
     'pete': 15359,
     'clements': 3743,
     'opportunity': 14584,
     'serving': 18672,
     'see': 18516,
     'error': 7109,
     'past': 15087,
     'behaviors': 1915,
     'arrives': 1148,
     'march': 12722,
     'bused': 2836,
     'penitentiary': 15236,
     'manhattan': 12664,
     'eight': 6631,
     'hour': 9990,
     'day': 5169,
     'returning': 17574,
     'night': 14096,
     'strict': 20155,
     'supervision': 20429,
     'furloughs': 8507,
     'allow': 623,
     'use': 22230,
     'skills': 19163,
     'betterment': 2048,
     'community': 4070,
     'chance': 3372,
     'added': 251,
     'request': 17399,
     'rnc': 17770,
     'deputy': 5564,
     'finance': 7930,
     'chairman': 3344,
     'denied': 5482,
     'environment': 7027,
     'easy': 6531,
     'backslide': 1565,
     'criminality': 4820,
     'grimacing': 9129,
     'clutching': 3828,
     'shoulder': 18928,
     'fox': 8307,
     'nfl': 14046,
     'announcer': 869,
     'joe': 11212,
     'buck': 2725,
     'tore': 21303,
     'rotator': 17894,
     'cuff': 4940,
     'awkward': 1495,
     'throw': 21110,
     'sideline': 19000,
     'quarter': 16571,
     'buccaneers': 2722,
     'vs': 22640,
     'cowboys': 4722,
     'game': 8584,
     'hate': 9505,
     'go': 8897,
     'especially': 7133,
     'routine': 17914,
     'erin': 7094,
     'field': 7872,
     'conditions': 4236,
     'thousand': 21085,
     'times': 21182,
     'commentator': 4022,
     'troy': 21619,
     'aikman': 484,
     'went': 22893,
     'hard': 9438,
     'stumbling': 20216,
     'first': 7980,
     'words': 23159,
     'sentence': 18613,
     'still': 20005,
     'ground': 9157,
     'writhing': 23233,
     'pain': 14894,
     'update': 22157,
     'mouth': 13606,
     'look': 12340,
     'right': 17705,
     'twisted': 21755,
     'awkwardly': 1496,
     'shock': 18885,
     'crossed': 4867,
     'face': 7535,
     'bad': 1578,
     'saw': 18251,
     'al': 521,
     'michaels': 13158,
     'tear': 20823,
     'acl': 199,
     'touchdown': 21339,
     'call': 2942,
     'way': 22803,
     'going': 8925,
     'announcing': 872,
     'least': 11941,
     'month': 13506,
     'treated': 21519,
     'concussion': 4225,
     'analyze': 790,
     'play': 15601,
     'conversion': 4515,
     'categorically': 3204,
     'denying': 5516,
     'allegations': 599,
     'tactic': 20664,
     'unconstitutional': 21859,
     'unfairly': 21952,
     'targeted': 20759,
     'players': 15606,
     'protested': 16315,
     'anthem': 900,
     'commissioner': 4036,
     'roger': 17822,
     'goodell': 8949,
     'released': 17214,
     'statement': 19914,
     'sunday': 20387,
     'defending': 5325,
     'subject': 20242,
     'panthers': 14957,
     'safety': 18074,
     'reid': 17158,
     'random': 16741,
     'stop': 20058,
     'frisk': 8408,
     'searches': 18457,
     'simply': 19076,
     'keep': 11441,
     'clean': 3720,
     'provide': 16333,
     'safe': 18065,
     'benefits': 1985,
     'case': 3153,
     'received': 16934,
     'anonymous': 882,
     'tip': 21203,
     'suspicious': 20521,
     'mask': 12807,
     'obscuring': 14361,
     'acting': 220,
     'aggressively': 443,
     'towards': 21363,
     'decided': 5242,
     'inform': 10588,
     'proper': 16255,
     'authorities': 1417,
     'conference': 4243,
     'advised': 360,
     'loitering': 12319,
     'line': 12184,
     'scrimmage': 18414,
     'sensitive': 18604,
     'areas': 1080,
     'avoid': 1468,
     'similar': 19061,
     'incidents': 10415,
     'moving': 13619,
     'forward': 8287,
     'described': 5585,
     'unidentified': 21987,
     'object': 14339,
     'hands': 9402,
     'description': 5588,
     'prompted': 16236,
     'officials': 14448,
     'detain': 5646,
     'perform': 15269,
     'thorough': 21076,
     'strip': 20170,
     'search': 18455,
     'relieved': 17230,
     'discover': 5938,
     'football': 8190,
     'single': 19095,
     'player': 15605,
     'code': 3871,
     'conduct': 4238,
     'teammates': 20821,
     'currently': 4983,
     'held': 9638,
     'questioning': 16594,
     'suspicion': 20519,
     'gang': 8591,
     'related': 17198,
     'activity': 230,
     'eyewitnesses': 7523,
     'observed': 14366,
     'wearing': 22825,
     'clothes': 3807,
     'bearing': 1850,
     'colors': 3965,
     'threatening': 21090,
     'logo': 12314,
     'quashing': 16576,
     'rumors': 17978,
     'team': 20820,
     'early': 6505,
     'exit': 7362,
     'las': 11829,
     'vegas': 22346,
     'oakland': 14321,
     'raiders': 16685,
     'announced': 866,
     'entirety': 6999,
     'home': 9864,
     'schedule': 18308,
     'head': 9545,
     'coach': 3837,
     'jon': 11238,
     'gruden': 9176,
     'backyard': 1573,
     'really': 16884,
     'perfect': 15267,
     'venue': 22380,
     'fact': 7551,
     'playing': 15609,
     'yard': 23291,
     'nowhere': 14253,
     'else': 6709,
     'league': 11920,
     'proposed': 16270,
     'half': 9339,
     'acre': 213,
     'plot': 15644,
     'nestled': 13990,
     'bay': 1823,
     'area': 1079,
     'suburbs': 20292,
     'boasted': 2309,
     'natural': 13889,
     'surface': 20464,
     'enough': 6949,
     'improvised': 10379,
     'seating': 18472,
     'accommodate': 137,
     'dozens': 6310,
     'hardcore': 9439,
     'faithful': 7596,
     'mistake': 13348,
     'rocking': 17812,
     'mean': 12942,
     'derek': 5570,
     'carr': 3118,
     'delivering': 5413,
     'strikes': 20165,
     'deck': 5253,
     'plus': 15665,
     'fans': 7640,
     'love': 12393,
     'amenities': 720,
     'room': 17865,
     'black': 2172,
     'hole': 9841,
     'garbage': 8602,
     'cans': 3027,
     'got': 8976,
     'bathrooms': 1799,
     'crockpot': 4854,
     'full': 8468,
     'chili': 3516,
     'house': 9992,
     'better': 2047,
     'spend': 19648,
     'several': 18699,
     'admitted': 306,
     'despite': 5618,
     'treacherous': 21508,
     'clothesline': 3808,
     'exposed': 7444,
     'tree': 21526,
     'roots': 17873,
     'far': 7644,
     'preferable': 15959,
     'games': 8587,
     'goddamn': 8909,
     'baseball': 1767,
     'humane': 10048,
     'deal': 5193,
     'suffering': 20331,
     'cleveland': 3748,
     'browns': 2691,
     'tuesday': 21666,
     'euthanized': 7211,
     'dawg': 5165,
     'pound': 15879,
     'rabies': 16640,
     'outbreak': 14712,
     'part': 15040,
     'heartbroken': 9582,
     'cutting': 5019,
     'lives': 12258,
     'short': 18911,
     'putting': 16511,
     'option': 14610,
     'owner': 14842,
     'jimmy': 11187,
     'haslam': 9492,
     'revealed': 17590,
     'concern': 4200,
     'piqued': 15523,
     'began': 1897,
     'chewing': 3493,
     'plastic': 15592,
     'seats': 18473,
     'salivating': 18117,
     'uncontrollably': 21861,
     'discovered': 5939,
     'late': 11840,
     'cure': 4973,
     'administered': 292,
     'put': 16497,
     'seemed': 18529,
     'fun': 8476,
     'approachable': 1016,
     'getting': 8765,
     'aggressive': 442,
     'bit': 2153,
     'seem': 18528,
     'quality': 16561,
     'life': 12125,
     'never': 14019,
     'battling': 1818,
     'constant': 4374,
     'seizures': 18548,
     'hydrophobia': 10121,
     'resulting': 17530,
     'making': 12598,
     'impossible': 10354,
     'drink': 6367,
     'beer': 1887,
     'emphasized': 6798,
     'sadness': 18064,
     'mercy': 13077,
     'killing': 11534,
     'assured': 1283,
     'comfort': 3997,
     'knowing': 11619,
     'suffer': 20329,
     'recognition': 16965,
     'brave': 2544,
     'altruistic': 679,
     'risk': 17745,
     'health': 9568,
     'greater': 9080,
     'good': 8947,
     'pentagon': 15246,
     'thursday': 21130,
     'honor': 9899,
     'sacrifices': 18050,
     'jerseys': 11155,
     'throughout': 21109,
     'december': 5234,
     'every': 7238,
     'week': 22851,
     'men': 13047,
     'gridiron': 9115,
     'bodies': 2320,
     'soldiers': 19433,
     'wear': 22823,
     'caps': 3059,
     'show': 18936,
     'support': 20440,
     'spokesperson': 19714,
     'amato': 692,
     'active': 223,
     'duty': 6472,
     'sporting': 19727,
     'gear': 8663,
     'teams': 20822,
     'raise': 16698,
     'awareness': 1489,
     'people': 15249,
     'aside': 1203,
     'preserve': 16008,
     'families': 7623,
     'travel': 21497,
     'cities': 3647,
     'across': 217,
     'uphold': 22170,
     'nation': 13868,
     'traditions': 21406,
     'battered': 1808,
     'bruised': 2696,
     'years': 23302,
     'often': 14458,
     'cut': 5013,
     'sit': 19123,
     'barracks': 1739,
     'enjoy': 6928,
     'freedom': 8359,
     'end': 6861,
     'service': 18668,
     'hopefully': 9925,
     'shows': 18946,
     'officers': 14442,
     'true': 21627,
     'heroes': 9700,
     'welling': 22885,
     'emotion': 6785,
     'upon': 22179,
     'finally': 7928,
     'setting': 18681,
     'foot': 8188,
     'hallowed': 9346,
     'tile': 21168,
     'college': 3941,
     'senior': 18591,
     'anthony': 901,
     'harper': 9470,
     'fulfilled': 8464,
     'lifelong': 12130,
     'dream': 6346,
     'saturday': 18229,
     'allowed': 626,
     'shower': 18942,
     'notre': 14231,
     'dame': 5078,
     'showers': 18943,
     'knew': 11603,
     'worked': 23165,
     'quit': 16620,
     'takes': 20696,
     'lather': 11844,
     'ovation': 14763,
     'brian': 2603,
     'kelly': 11451,
     'tossed': 21326,
     'conditioner': 4234,
     'bench': 1969,
     'locker': 12296,
     'watch': 22781,
     'wishing': 23092,
     'always': 684,
     'thought': 21080,
     'soap': 19382,
     'goes': 8918,
     'grit': 9143,
     'determination': 5667,
     'anyone': 930,
     'achieve': 186,
     'bathe': 1794,
     'entire': 6997,
     'witness': 23112,
     'announcement': 867,
     'perceived': 15258,
     'major': 12587,
     'reassurance': 16906,
     'parents': 15012,
     'children': 3514,
     'low': 12400,
     'cognitive': 3889,
     'abilities': 49,
     'subpar': 20258,
     'reasoning': 16901,
     'pediatric': 15197,
     'experts': 7403,
     'report': 17358,
     'claims': 3667,
     'contact': 4413,
     'poses': 15824,
     'little': 12246,
     'brains': 2521,
     'already': 656,
     'well': 22881,
     'tackle': 20661,
     'long': 12331,
     'known': 11622,
     'high': 9742,
     'sport': 19725,
     'particularly': 15050,
     'poor': 15770,
     'guys': 9280,
     'knuckle': 11624,
     'draggers': 6320,
     'away': 1490,
     'lose': 12367,
     'university': 22026,
     'chicago': 3497,
     'childhood': 3512,
     'development': 5701,
     'expert': 7402,
     'dr': 6313,
     'maureen': 12883,
     'clifford': 3759,
     'neuropathological': 14006,
     'research': 17418,
     'led': 11953,
     'conclusion': 4215,
     'chronic': 3592,
     'traumatic': 21495,
     'encephalopathy': 6839,
     'caused': 3226,
     'repeated': 17334,
     'severe': 18700,
     'impacts': 10296,
     'mitigated': 13362,
     'percent': 15259,
     'cases': 3154,
     'youth': 23362,
     'presented': 16004,
     'signs': 19042,
     'huge': 10036,
     'dumbass': 6436,
     'clearly': 3737,
     'couple': 4679,
     'screws': 18411,
     'loose': 12349,
     'course': 4691,
     'cte': 4927,
     'danger': 5098,
     'comes': 3996,
     'sports': 19728,
     'ages': 435,
     'crucial': 4885,
     'healthy': 9570,
     'neurological': 14004,
     'growth': 9174,
     'symptoms': 20621,
     'mood': 13514,
     'swings': 20584,
     'difficult': 5800,
     'thinking': 21053,
     'memory': 13046,
     'sounds': 19514,
     'kid': 11514,
     'precious': 15933,
     'dude': 6424,
     'bonehead': 2379,
     'blocking': 2256,
     'tackling': 20662,
     'hit': 9805,
     'crossing': 4870,
     'routes': 17913,
     'reasons': 16902,
     'idiot': 10187,
     'study': 20209,
     'concluded': 4213,
     'halfwits': 9341,
     'shot': 18924,
     'success': 20299,
     'staring': 19888,
     'wide': 22993,
     'eyed': 7516,
     'table': 20649,
     'unopened': 22057,
     'presents': 16007,
     'largely': 11822,
     'ignored': 10213,
     'guests': 9225,
     'local': 12287,
     'rick': 17681,
     'joseph': 11251,
     'reportedly': 17361,
     'watched': 22783,
     'helplessly': 9659,
     'white': 22964,
     'elephant': 6678,
     'exchange': 7298,
     'devolved': 5720,
     'friends': 8394,
     'chatting': 3437,
     'nice': 14058,
     'christ': 3579,
     'turn': 21707,
     'pick': 15446,
     'ago': 450,
     'derailed': 5567,
     'everyone': 7241,
     'blabbing': 2171,
     'fucking': 8450,
     'christmas': 3588,
     'forced': 8206,
     'listen': 12224,
     'engaged': 6900,
     'pleasant': 15618,
     'conversations': 4513,
     'favorite': 7721,
     'recipes': 16945,
     'beloved': 1961,
     'memories': 13044,
     'festive': 7842,
     'season': 18465,
     'ugh': 21788,
     'disaster': 5905,
     'chumps': 3597,
     'strategizing': 20120,
     'screw': 18409,
     'best': 2030,
     'instead': 10728,
     'wasting': 22779,
     'bond': 2375,
     'fighting': 7891,
     ...}



# Pair: 

For a final exercise, work through in pairs the following exercise.

Create a document term matrix of the 1000 document corpus.  The vocabulary should have no stopwords, no numbers, no punctuation, and be lemmatized.  The Document-Term Matrix should be created using tfidf.


```python
#__SOLUTION__
corpus = pd.read_csv('data/satire_nosatire.csv')

```


```python
#__SOLUTION__
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
#__SOLUTION__
docs = [doc_preparer(doc) for doc in corpus.body]
```


```python
#__SOLUTION__
tf_idf = TfidfVectorizer(min_df = .05)
X = tf_idf.fit_transform(docs)

df = pd.DataFrame(X.toarray())
df.columns = tf_idf.vocabulary_
df.head()
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
      <th>secretary</th>
      <th>mark</th>
      <th>third</th>
      <th>top</th>
      <th>administration</th>
      <th>official</th>
      <th>less</th>
      <th>three</th>
      <th>week</th>
      <th>tell</th>
      <th>...</th>
      <th>reserve</th>
      <th>economy</th>
      <th>october</th>
      <th>militant</th>
      <th>reporting</th>
      <th>seven</th>
      <th>syria</th>
      <th>dec</th>
      <th>minimum</th>
      <th>lawmaker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.162442</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.078375</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.106884</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.13159</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.208572</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.077942</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.061257</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.128578</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.149827</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 658 columns</p>
</div>




```python

```
