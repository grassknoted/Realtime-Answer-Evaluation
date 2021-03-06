import time	# To calculate execution times

prgStart = time.time()	# Start Timer

# Importing Keras modules for Neural Network
from keras import backend
from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

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

# Miscellaneous Imports
import csv
import os

# Importing Graphs File:
from Graphs import *

# Import Evaluation Metric File:
from EvaluationMetric import *



'''
Parameters to Test Models:
'''

# File paths to load:
# Training Set:
training_set_path = "/media/hduser/OS_Install/Mechanical Engineering/Sem V/Data Analytics/Project/FinalTrain.csv"

# Testing Set:
testing_set_path = "/media/hduser/OS_Install/Mechanical Engineering/Sem V/Data Analytics/Project/FinalTest.csv"

# Toggle stemming: 1 -> Enable; 0 -> Disable
stemming = 0

# Toggle spell correction: 1 -> Enable; 0 -> Disable
spell_correction = 0

# Sets to consider for Training
sets_to_train_with = [1, 2, 3, 4, 5, 6, 7, 8]   # Sets 1 - 8 (All Sets)

#sets_to_train_with = [1, 2, 3, 4, 5, 6]         # Sets 1 - 6

#sets_to_train_with = [3, 4, 5, 6]               # Sets 3 - 6

#sets_to_train_with = [7, 8]                     # Sets 7 - 8

#sets_to_train_with = [1]                         # Set 1

# Word Vector dimensions
vector_dimensions = 200

# Word2Vec model Training Window
window_size = 5

# Number of workers used to create the Word2Vec model
number_of_workers = 2

# Minimum number of times a word must be present to be
# included in the Word2Vec model
min_word_count = 5

# Specify name of the Word2Vec model to load
# If no name is specified, a model is built
Word2VecModelName = ""

# Number of epochs for Neural Network Training
training_epochs = 20

# Training batch size
training_batch_size  = 10

# Toggle normalization of labels: 1 -> Enable; 0 -> Disable
normalize_labels = 0

# Set the word limit for each essay
max_length = 500



'''
Loading the GloVe Model
'''
#Declaring a dictionary to contain a mapping for each WordVector
embeddings_index = dict()

#Opening the file with the pre-trained GloVe Vectors
f = open('glove.6B.200d.txt')

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
Loading of Training and Testing Data:
  training_set  = dataframe of training data
  testing_set  = dataframe of testing data 
'''
print("Loading files..")

# Initialize file_load timer
file_load_start = time.time()
training_set = pd.read_csv(training_set_path, header = 0, delimiter = "\t", quoting = 3)

testing_set = pd.read_csv(testing_set_path, header = 0, delimiter = "\t", quoting = 3)

# End and display file load timer
print("Files loaded in : ",time.time()-file_load_start,"s.\n")



# Loading punkt tokenizer
#tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#Loading keras tokenizer
t = Tokenizer()
t.fit_on_texts(training_set["essay"])
word_index = t.word_index
vocab_size = len(t.word_index) + 1



'''
Preprocessing data to create Word2Vec Model
'''
# Initialize empty lists
training_sentences = []
training_answer_data = []
testing_sentences = []
testing_answer_data = []	

# Training Set Pre-processing
print("Training set processing..")

# Initialize Train process
start = time.time()

for i in sets_to_train_with:
    # Consider set by set
    set_in_consideration = training_set[training_set["essay_set"] == i]
    
    for answer_in_consideration in set_in_consideration["essay"]:
        # Append the essays
        training_answer_data.append(answer_in_consideration)

# End and display time to process Train set
print("Processing Done in : ",time.time()-start,"s.\n")

# Testing Set Pre-processing
print("Testing set processing..")

# Initialize Test process timer
start = time.time()

for i in sets_to_train_with:
     # Consider set by set
    set_in_consideration = testing_set[testing_set["essay_set"] == i]

    for answer_in_consideration in set_in_consideration["essay"]:
        # Append the essays
        testing_answer_data.append(answer_in_consideration)

# End and display time to process Test set
print("Processing Done in : ",time.time()-start,"s.\n")



'''
Extract Train Labels set by set
'''
train_labels = []
test_labels = []

# If the labels are to be normalized
if(normalize_labels):

    # Labels normalized to form classes: 0 - 3
    for i in sets_to_train_with:
        # Traverse set by set
        set_in_consideration = training_set[training_set["essay_set"] == i]
        
        # Append domain1_score as the label
        for domain1_score in set_in_consideration["domain1_score"]:
            if(i==1):
                train_labels.append(round(domain1_score*0.25))
            elif(i==2):
                train_labels.append(round(domain1_score*0.5))
            elif(i==3):
                train_labels.append(round(domain1_score*1.0))
            elif(i==4):
                train_labels.append(round(domain1_score*1.0))
            elif(i==5):
                train_labels.append(round(domain1_score*0.75))
            elif(i==6):
                train_labels.append(round(domain1_score*0.75))
            elif(i==7):
                train_labels.append(round(domain1_score*0.1))
            elif(i==8):
                train_labels.append(round(domain1_score*0.05))

    for i in sets_to_train_with:
        # Traverse set by set
        set_in_consideration = testing_set[testing_set["essay_set"] == i]

        # Append domain1_score as the label
        for domain1_score in set_in_consideration["domain1_score"]:
            if(i==1):
                test_labels.append(round(domain1_score*0.25))
            elif(i==2):
                test_labels.append(round(domain1_score*0.5))
            elif(i==3):
                test_labels.append(round(domain1_score*1.0))
            elif(i==4):
                test_labels.append(round(domain1_score*1.0))
            elif(i==5):
                test_labels.append(round(domain1_score*0.75))
            elif(i==6):
                test_labels.append(round(domain1_score*0.75))
            elif(i==7):
                test_labels.append(round(domain1_score*0.1))
            elif(i==8):
                test_labels.append(round(domain1_score*0.05))

else:

    for i in sets_to_train_with:
        # Traverse set by set
        set_in_consideration = training_set[training_set["essay_set"] == i]
        
        # Append domain1_score as the label
        for domain1_score in set_in_consideration["domain1_score"]:
            if(i==1):
                train_labels.append(round(domain1_score))
            elif(i==2):
                train_labels.append(round(domain1_score))
            elif(i==3):
                train_labels.append(round(domain1_score))
            elif(i==4):
                train_labels.append(round(domain1_score))
            elif(i==5):
                train_labels.append(round(domain1_score))
            elif(i==6):
                train_labels.append(round(domain1_score))
            elif(i==7):
                train_labels.append(round(domain1_score))
            elif(i==8):
                train_labels.append(round(domain1_score))
    
    for i in sets_to_train_with:
        # Traverse set by set
        set_in_consideration = testing_set[testing_set["essay_set"] == i]

        # Append domain1_score as the label
        for domain1_score in set_in_consideration["domain1_score"]:
            if(i==1):
                test_labels.append(round(domain1_score))
            elif(i==2):
                test_labels.append(round(domain1_score))
            elif(i==3):
                test_labels.append(round(domain1_score))
            elif(i==4):
                test_labels.append(round(domain1_score))
            elif(i==5):
                test_labels.append(round(domain1_score))
            elif(i==6):
                test_labels.append(round(domain1_score))
            elif(i==7):
                test_labels.append(round(domain1_score))
            elif(i==8):
                test_labels.append(round(domain1_score))



'''
Preparing input into the LSTM Model
'''
# Encoding and Padding the Training Essays
encoded_train = t.texts_to_sequences(training_answer_data)
padded_train = pad_sequences(encoded_train, maxlen=max_length, padding='post')

# Encoding and Padding the Testing Essays
encoded_test = t.texts_to_sequences(testing_answer_data)
padded_test = pad_sequences(encoded_test, maxlen=max_length, padding='post')

#Loading the Train and Test Labels as a numpy arrays
train_labels = np.asarray(train_labels)
test_labels = np.asarray(test_labels) 

# Initializing the Embedding Matrix to be passed to the LSTM
embedding_matrix = np.zeros((vocab_size, vector_dimensions))

# Looping through each unique word among all esaays
for word, i in t.word_index.items():

    #Loading the vector of that word
    embedding_vector = embeddings_index.get(word)

    #Testing to see if that word was loaded into the GloVe Model and adding it if it is
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

train_labels = np.asarray(train_labels)
test_labels = np.asarray(test_labels)



'''
Building the LSTM Model
'''
# Building the LSTM Model
print("Building the LSTM Model ..")

# Initialize Neural Network Building timer
s = time.time()

# Set seed = 9 to ensure reproducibility
np.random.seed(9)

# Use a sequential network architecture
model = Sequential()

# Embedding layer: 300 Input Dimension
model.add(Embedding(vocab_size, vector_dimensions, weights=[embedding_matrix], trainable=False))

# Input Layer: 128 LSTM Neurons
model.add(LSTM(107, dropout=0.2, recurrent_dropout=0.2))

# Output Layer: Output = 61, SoftMax Funtion
model.add(Dense(61, activation='softmax'))

#Compile the LSTM
model.compile(loss= 'sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# LSTM Model building is complete
print("LSTM Model built in : ",time.time()-s,"s.\n")



'''
Training the LSTM Model
'''
# Begin training, initialize Training Timer
print("Training the LSTM..")
s = time.time()

print("Train Lables: ", len(train_labels))

# Fit the model on the training data
model.fit(padded_train, train_labels, epochs = training_epochs)

# Evaluation of the Model
scores = model.evaluate(padded_test, test_labels)

# Print log messages
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# End training, and print training time
print("LSTM Model trained in : ",time.time()-s,"s.\n")



# Testing the Model
prediction = model.predict(padded_test)

# Getting the value of the class from SoftMax layer
actual_predictions = []

# Find position at which the maximum value occured in the vector
# the same purpose as argmax()
for i in range(0,len(prediction)):
    # Set maxi to a low value of -9 initially
    maxi = -9
    pos = 0

    # Traverse the output of the SoftMax Layer
    for j in range(0, len(prediction[i])):

        # If value > current_maximum
        if(prediction[i][j] > maxi):
            # current_maximum = value
            maxi = prediction[i][j]
            # Set position to position where max occured
            pos = j

    # Append true value of score
    actual_predictions.append(pos)

# Compute Quadratic Weighted Kappa for test labels
qwk_score = quadratic_weighted_kappa(np.asarray(test_labels), actual_predictions)

# Display Quadratic Weighter Kappa Value for current model
print("Quadratic Weighted Kappa with LSTM Model: ", qwk_score, "\n\n\n")

# Plotting graph
plot_error(test_labels, actual_predictions, str("GloVe LSTM with "+str(vector_dimensions)+"D Vectors"))