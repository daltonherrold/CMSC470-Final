# Developed from Code from Alex Jian Zheng
import random
import math
import torch
import torch.nn as nn
import numpy as np
from numpy import zeros, sign
from math import exp, log
from collections import defaultdict
import json

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import argparse

class Example:
    """
    Class to represent a logistic regression example
    """
    def __init__(self, json_line, vocab, use_bias=True):
        """
        Create a new example

        json_line -- The json object that contains the label ("label") and features as fields
        vocab -- The vocabulary to use as features (list)
        use_bias -- Include a bias feature (should be false with Pytorch)
        """

        # Use but don't modify this function
        
        self.nonzero = {}
        self.y = 1 if json_line["label"] else 0
        self.x = zeros(len(vocab))

        for feature in json_line:
            if feature in vocab:
                assert feature != 'BIAS_CONSTANT', "Bias can't actually appear in document"
                self.x[vocab.index(feature)] += float(json_line[feature])
                self.nonzero[vocab.index(feature)] = feature
        # Initialize the bias feature
        if use_bias:
            self.x[0] = 1


class GuessDataset(Dataset):
    def __init__(self, vocab):
        self.vocab = vocab
        
        # Just create some dummy data so unit tests will fail rather than cause error
        self.num_features = len(vocab)
        self.feature = zeros((5, self.num_features))
        self.label = zeros((5, 1))
        self.num_samples = 5

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.num_samples

    def initialize(self, filename):
        # Complete this function to actually populate the feature and label members of the class with non-zero data.
        dataset = []                                                 #sol
        with open(filename) as infile:                               #sol
            for line in infile:                                      #sol
                ex = Example(json.loads(line), self.vocab, use_bias=False)#sol
                dataset.append(ex)                                   #sol

        # You may want to use numpy's fromiter function
        
        features = np.stack(list(ex.x for ex in dataset))            #sol
        label = np.stack(list(np.array([ex.y]) for ex in dataset))   #sol

        self.feature = torch.from_numpy(features.astype(np.float32)) #sol
        self.label = torch.from_numpy(label.astype(np.float32))      #sol
        self.num_samples = len(self.label)                            #sol
        assert self.num_samples == len(self.feature)
        None         

class LogRegBuzzer(nn.Module):
    def __init__(self, num_features):
        """
        Initialize the parameters you'll need for the model.

        :param num_features: The number of features in the linear model
        """
        super(LogRegBuzzer, self).__init__()
        # Replace this with a real nn.Module
        self.linear = None
        self.linear = nn.Linear(num_features, 1)                          #sol

    def forward(self, x):
        """
        Compute the model prediction for an example.

        :param x: Example to evaluate
        """
        y_pred = torch.sigmoid(self.linear(x))                            #sol Label has only two categories, so sigmoid and softmax should be essentially the same.
        return y_pred                                                     #sol
        return 0.5

    def evaluate(self, data):
        """
        Computes the accuracy of the model. 
        """

        # No need to modify this function.
        with torch.no_grad():
            y_predicted = self(data.feature)
            y_predicted_cls = y_predicted.round()
            acc = y_predicted_cls.eq(data.label).sum() / float(data.label.shape[0])
            return acc

def step(epoch, ex, model, optimizer, criterion, inputs, labels):
    y_pred = model(inputs)
    loss = criterion(y_pred, labels)
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()


def train_and_save():
    print('Training pytorch.')
    with open('vocab', 'r') as infile:
        vocab = [x.strip() for x in infile]  

    train = GuessDataset(vocab)

    train.initialize('log_reg_training.json')

    logreg = LogRegBuzzer(train.num_features)

    num_epochs = 100
    batch = 1

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(logreg.parameters(), lr=0.1)
    
    train_loader = DataLoader(dataset=train,
                              batch_size=batch,
                              shuffle=True,
                              num_workers=0)

    for epoch in range(num_epochs):
      for ex, (inputs, labels) in enumerate(train_loader):
        step(epoch, ex, logreg, optimizer, criterion, inputs, labels)
    
    print('Saving pytorch model.')
    
    torch.save(logreg.state_dict(), './trained_model.th')
