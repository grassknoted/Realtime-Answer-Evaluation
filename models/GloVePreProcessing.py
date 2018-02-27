import time	# To calculate execution times

prgStart = time.time()	# Start Timer

# Importing Word2Vec for Word Embeddings
from gensim.models import Word2Vec
from gensim.models import word2vec

# Importing NLP tools: NLTK
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer

# Importing the Keras Tokenizer
from keras.preprocessing.text import Tokenizer

# Importing data from NLTK
import nltk.data

# Import Pandas for dataframe manipulation
import pandas as pd

# Import logging to log model building progress
import logging

# Import NumPy
import numpy as np

# Importing packages required for Spell Checking
import re
from collections import Counter

# Miscellaneous Imports
import csv
import os



'''
Parameters to Test Models:
'''

# File paths to load:
# WORDS Vocabulary Training Path:
WORDS_path = "/media/hduser/OS_Install/Mechanical Engineering/Sem V/Data Analytics/Project/FinalTrain.csv"

# Toggle stemming: 1 -> Enable; 0 -> Disable
stemming = 0

# Toggle spell correction: 1 -> Enable; 0 -> Disable
spell_correction = 0



#Loading keras tokenizer
t = Tokenizer()



'''
Loading the GloVe Model
'''
#Declaring a dictionary to contain a mapping for each WordVector
embeddings_index = dict()

#Opening the file with the pre-trained GloVe Vectors
f = open('glove.6B.100d.txt')

#Pasring through the GloVe File line by line
for line in f:
    
    #Getting a list of each word in a line
    values = line.split()

    #Loading the each word Vector into the Dictionary
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

#Closing the File
f.close()

#Printing the Loading status
print('\nLoaded %s word vectors\n' % len(embeddings_index))



'''
Function to create word vectors
'''
def makeFeatureVectors(words, num_features):
    '''
    Create a word vectors of the words passed using the Word2Vec Model
    '''
    #Tokenizing the words sent to the fucntion
    t.fit_on_texts(words)
    word_index = t.word_index
    vocab_size = len(t.word_index) + 1

    # Initial word vector to a zero vector
    featureVec = np.zeros((num_features,), dtype='float32')

    # Initialize Number of Words to 0
    nwords = 0

    #Building the Feature Vector one word at a time
    for word in words:
        #Getting the Word Vector for the given word
        embedding_vector = embeddings_index.get(word)
        
        #Checking if a Word Vector for that word exists
        if embedding_vector is not None:
            # Count the number of words
            nwords = nwords + 1
            # Create a vector by adding all word vectors it contains
            featureVec = np.add(featureVec,embedding_vector)
    
    # Divide feature vector by 0 to get average word vector
    featureVec = np.divide(featureVec,nwords)

    return featureVec



'''
Function to create word vectors from a set of answers
'''
def getAvgFeatureVecs(answers, num_features):
    '''
    Given a set of answers, calculate the average feature 
    vector and return a 2D numpy array for each one 
    '''
    # Initialize a counter
    counter = 0

    # Allocate a 2D numpy array, for speed
    answerFeatureVecs = np.zeros((len(answers),num_features),dtype="float32")
    
    # Loop through the answers in an answer set
    for answer in answers:

       # Print a status message every 1000th answer
       if (counter%1000 == 0):
           print("Answer %d of %d processed." % (counter, len(answers)))
       
       # Get average feature vector for each answer in answer set
       answerFeatureVecs[counter] = makeFeatureVectors(answer, num_features)
       
       # Increment the counter
       counter = counter + 1

    return answerFeatureVecs



'''
Spell Checker:
'''
def words(text):
    "Returns all words of `text` in lowercase"
    return re.findall(r'\w+', text.lower())

# Word Count of all words in the specified file
WORDS = Counter(words(open(WORDS_path).read()))

# Stemming of Words
stemmer = PorterStemmer()

def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N

def correction(word):
    "Most probable spelling correction for `word`"
    return max(candidates(word), key=P)

def candidates(word):
    "Generate possible spelling corrections for `word`"
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    '''
    All edits that are one edit away from `word`.
    '''
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`"
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))



'''
Convert `raw_answer` to a list of words:
'''
def answer_to_wordlist(raw_answer):
    '''
    The answer is converted to a list of meaningful words
    '''
    # Removing all numbers
    letters_only = re.sub("[^a-zA-Z]", " ", raw_answer)

    # Converting all letters to lowercase
    words = letters_only.lower().strip().split()

    # Creating a set of stopwords
    stops = set(stopwords.words("english"))

    # Remove stopwords
    meaningful_words = [w for w in words if not w in stops] 

    # Stemming
    if(stemming):
        meaningful_words = [stemmer.stem(w) for w in words if not w in meaningful_words]

    # Spell correction:
    if(spell_correction):
        meaningful_words = [stemmer.stem(w) for w in words if not w in meaningful_words]

    return meaningful_words



'''
Convert `answer` into sentences
'''
def answer_to_sentences(answer, tokenizer, remove_stopwords = False):
    '''
    The answer is tokenized to form a list of sentences
    '''
    # Splitting each paragraph into sentences
    raw_sentences = tokenizer.tokenize(answer.strip())

    # Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:

        # If a sentence is empty, skip it
        if(len(raw_sentence) > 0):
            sentences.append(answer_to_wordlist(raw_sentence))

    return sentences


