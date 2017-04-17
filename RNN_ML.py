# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 19:12:20 2017

@author: admin
"""


import itertools
import numpy as np
import nltk
import sys
from datetime import datetime
from itertools import islice

english= open ("fr-en/europarl-v7.fr-en.en","r")
french= open ("fr-en/europarl-v7.fr-en.fr","r")
en_head = list(islice(english, 100))
fr_head= list(islice(french, 100))


sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

en_words=[]
fr_words=[]
en_sentences = itertools.chain(*[nltk.sent_tokenize(x.lower()) for x in en_head])
en_sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in en_sentences]
fr_sentences = itertools.chain(*[nltk.sent_tokenize(x.lower(), language = "French") for x in fr_head])
fr_sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in fr_sentences]

en_words = [nltk.word_tokenize(sent) for sent in en_sentences]
fr_words = [nltk.word_tokenize(sent) for sent in fr_sentences]


unknown_token = "UNKNOWN_TOKEN"
en_vocab_size=10000

# Count the word frequencies
en_word_freq = nltk.FreqDist(itertools.chain(*en_words))
len(en_word_freq.items())
fr_word_freq = nltk.FreqDist(itertools.chain(*fr_words))
len(fr_word_freq.items())


en_vocab = en_word_freq.most_common(en_vocab_size-1)
fr_vocab = fr_word_freq.most_common(en_vocab_size-1)
en_index_to_word = [x[0] for x in en_vocab]
fr_index_to_word = [x[0] for x in fr_vocab]
en_index_to_word.append(unknown_token)
fr_index_to_word.append(unknown_token)
en_word_to_index = dict([(w,i) for i,w in enumerate(en_index_to_word)])
fr_word_to_index = dict([(w,i) for i,w in enumerate(fr_index_to_word)])

#SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
for i, sent in enumerate(en_words):
    en_words[i] = [w if w in en_word_to_index else unknown_token for w in sent]
for i, sent in enumerate(fr_words):
    fr_words[i] = [w if w in fr_word_to_index else unknown_token for w in sent]
#SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS

X_train = np.asarray([[en_word_to_index[w] for w in sent[:-1]] for sent in en_words])
y_train = np.asarray([[fr_word_to_index[w] for w in sent[:-1]] for sent in fr_words])


#SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
#len(y_train) = 103 but len(X_train) = 107??
X_train = X_train[0:103] 
for i in np.arange(103):
    max_size = max(len(X_train[i]), len(y_train[i]))
    if max_size > len(X_train[i]): 
        for j in np.arange(max_size - len(X_train[i])):
            X_train[i].append(en_word_to_index["UNKNOWN_TOKEN"])
        continue;
    if max_size > len(y_train[i]): 
        for j in np.arange(max_size - len(y_train[i])):
            y_train[i].append(fr_word_to_index["UNKNOWN_TOKEN"])
        continue;
#now all the vectors in the both sets should have the same size    
#SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS

class RNNNumpy:
     
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

npa = np.array
def softmax(w, t = 1.0):
    e = np.exp(npa(w) / t)
    dist = e / np.sum(e)
    return dist

def forward_propagation(self, x):
    T = len(x)
    s = np.zeros((T + 1, self.hidden_dim))
    s[-1] = np.zeros(self.hidden_dim)
    o = np.zeros((T, self.word_dim))
    for t in np.arange(T):
        s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
        o[t] = softmax(self.V.dot(s[t]))
    return [o, s]

RNNNumpy.forward_propagation = forward_propagation

def predict(self, x):
    o, s = self.forward_propagation(x)
    return np.argmax(o, axis=1)
RNNNumpy.predict = predict

#############################
#running an experiment#######
np.random.seed(10)
model = RNNNumpy(en_vocab_size)
o, s = model.forward_propagation(X_train[1])

predictions = model.predict(X_train[1])
predictions.shape
predictions

##############################

def calculate_total_loss(self, x, y):
    L = 0
    # For each sentence...
    for i in np.arange(len(y)):
        o, s = self.forward_propagation(x[i])
        # We only care about our prediction of the "correct" words
        correct_word_predictions = o[np.arange(len(y[i])), y[i]]
        # Add to the loss based on how off we were
        L += -1 * np.sum(np.log(correct_word_predictions))
    return L
 
def calculate_loss(self, x, y):
    # Divide the total loss by the number of training examples
    N = np.sum((len(y_i) for y_i in y))
    return self.calculate_total_loss(x,y)/N
 
RNNNumpy.calculate_total_loss = calculate_total_loss
RNNNumpy.calculate_loss = calculate_loss

########################################
###########continue with experiment#####
model.calculate_loss(X_train[:22], y_train[:22])
#######################################
def bptt(self, x, y):
    T = len(y)
    # Perform forward propagation
    o, s = self.forward_propagation(x)
    # We accumulate the gradients in these variables
    dLdU = np.zeros(self.U.shape)
    dLdV = np.zeros(self.V.shape)
    dLdW = np.zeros(self.W.shape)
    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1.
    # For each output backwards...
    for t in np.arange(T)[::-1]:
        dLdV += np.outer(delta_o[t], s[t].T)
        # Initial delta calculation
        delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
        # Backpropagation through time (for at most self.bptt_truncate steps)
        for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
            # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            dLdW += np.outer(delta_t, s[bptt_step-1])              
            dLdU[:,x[bptt_step]] += delta_t
            # Update delta for next step
            delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
    return [dLdU, dLdV, dLdW]
 
RNNNumpy.bptt = bptt

# Performs one step of SGD.
def numpy_sdg_step(self, x, y, learning_rate):
    # Calculate the gradients
    dLdU, dLdV, dLdW = self.bptt(x, y)
    # Change parameters according to gradients and learning rate
    self.U -= learning_rate * dLdU
    self.V -= learning_rate * dLdV
    self.W -= learning_rate * dLdW

RNNNumpy.sgd_step = numpy_sdg_step

def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print ("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print ("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

#the final round of test
np.random.seed(10)
# Train on a small subset of the data to see what happens
model = RNNNumpy(en_vocab_size)
losses = train_with_sgd(model, X_train[:20], y_train[:20], nepoch=10, evaluate_loss_after=1)
