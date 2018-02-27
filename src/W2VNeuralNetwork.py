import time	# To calculate execution times

prgStart = time.time()	# Start Timer

# Importing Keras modules for Neural Network
from keras import backend
from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense

# Importing Word2Vec for Word Embeddings
from gensim.models import Word2Vec
from gensim.models import word2vec

# Import logging to log model building progress
import logging

# Import NumPy
import numpy as np

# Miscellaneous Imports
import csv
import os

# Importing Graphs File:
from Graphs import *

# Import PreProcessing File:
from PreProcessing import *

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

#sets_to_train_with = [1]                         #Set 1

# Word Vector dimensions
vector_dimensions = 100

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
training_epochs = 1

# Training batch size
training_batch_size  = 10

# Toggle normalization of labels: 1 -> Enable; 0 -> Disable
normalize_labels = 0



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
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



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
        # Append the sentences
        training_sentences += answer_to_sentences(answer_in_consideration, tokenizer, True)
        # Append list of words
        training_answer_data.append(answer_to_wordlist(answer_in_consideration))

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
        # Append the sentences
        testing_sentences += answer_to_sentences(answer_in_consideration, tokenizer, True)
        # Append list of words
        testing_answer_data.append(answer_to_wordlist(answer_in_consideration))

# End and display time to process Test set
print("Processing Done in : ",time.time()-start,"s.\n")



'''
Building Word2Vector Model for Word Embeddings
'''
print("Building Word2Vec model..")

# Initialize Model Building timer
start = time.time()

# Check if a Word2Vec Model name is specified
if(Word2VecModelName):
    # Load a locally saved model
    v2wmodel = Word2Vec.load(Word2VecModelName)
else:
    # Building the Word2Vec Model with the specified parameters
    v2wmodel = word2vec.Word2Vec(training_sentences, size=vector_dimensions, window=window_size, min_count=min_word_count, workers=number_of_workers)
    
    # Save Word2Vec model with specified Name
    v2wmodel.save("Word2VecModel")

# End and display time to build Word2Vec Model
print("Model built in : ", time.time()-start,"s.\n")



'''
Embedding of Train Vectors
'''
print("Creating Embedded Train Vectors..")
start = time.time()

# Get the answerTrainVectors using a Average Word Vectors
answerTrainVectors = getAvgFeatureVecs(training_answer_data, v2wmodel, vector_dimensions)

print("Embedded Train Vectors created in : ", time.time()-start,"s.\n")



'''
Begin Embedding of Test Vectors
'''
print("Creating Embedded Test Vectors..")
start = time.time()

# Get the answerTestVectors using a Average Word Vectors
answerTestVectors = getAvgFeatureVecs(testing_answer_data, v2wmodel, vector_dimensions)

print("Embedded Test Vectors created in : ", time.time()-start,"s.\n")



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
Building the Neural Network Model
'''
#Building the Neural Network
print("Building the Neural Network..")

# Initialize Neural Network Building timer
s = time.time()

# Set seed = 9 to ensure reproducibility
np.random.seed(9)

# Begin Neural Network `model`
#, kernel_regularizer=regularizers.l2(0.0001), activity_regularizer=regularizers.l1(0.0001))

# Use a sequential network architecture
model = Sequential()

# Input Layer: 300 Input Neurons, Output = 107, Activation = Rectified Linear Units
model.add(Dense(output_dim = 80, input_dim = vector_dimensions, activation = 'relu'))

# Output Layer: 107 Input Neurons, Output = 61, SoftMax Function
model.add(Dense(output_dim = 61, input_dim = 80, activation = 'softmax'))

# Define Stochastic Gradient Descent
# REMOVE training_batch_size
#sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

# Compile the Neural Network
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'sgd',metrics=[metrics.mae, metrics.categorical_accuracy, 'accuracy'])
#model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = [metrics.mae])
#model.compile(loss='mean_squared_error',optimizer='sgd',metrics=[metrics.mae, metrics.categorical_accuracy])

# Neural Network building is complete
print("Neural Network built in : ",time.time()-s,"s.\n")



'''
Training the Neural Network Model
'''
# Begin training, initialize Training Timer
print("Training the Neural Network..")
s = time.time()

print("Train Lables: ", len(train_labels))

# Fit the model on the training data
model.fit(answerTrainVectors, np.asarray(train_labels), epochs = training_epochs, batch_size = training_batch_size)
#model.fit(tfidfTrainVectors, np.asarray(train_labels), epochs = training_epochs, batch_size = training_batch_size)

# Evaluation of the Model
scores = model.evaluate(answerTrainVectors, np.asarray(train_labels))

# Print log messages
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# End training, and print training time
print("Neural Network trained in : ",time.time()-s,"s.\n")

# Testing the Model
prediction = model.predict(answerTestVectors)

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
print("Quadratic Weighted Kappa with Deep Neural Network: ", qwk_score, "\n\n\n")

# Plotting graph
plot_error(test_labels, actual_predictions, str("Word2Vec Neural Network with "+str(vector_dimensions)+"D Vectors"))