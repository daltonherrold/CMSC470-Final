import torch
import torch.nn as nn
import numpy as np
from numpy import zeros
import json

from torch.utils.data import Dataset, DataLoader

class Example:
    def __init__(self, json_line, vocab, use_bias=True):
        self.nonzero = {}
        self.y = 1 if json_line["label"] else 0
        self.x = zeros(len(vocab))

        for feature in json_line:
            if feature in vocab:
                assert feature != 'BIAS_CONSTANT', "Bias can't actually appear in document"
                self.x[vocab.index(feature)] += float(json_line[feature])
                self.nonzero[vocab.index(feature)] = feature
        if use_bias:
            self.x[0] = 1


class GuessDataset(Dataset):
    def __init__(self, vocab):
        self.vocab = vocab
        self.num_features = len(vocab)
        self.feature = zeros((5, self.num_features))
        self.label = zeros((5, 1))
        self.num_samples = 5

    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    def __len__(self):
        return self.num_samples

    def initialize(self, filename):
        dataset = []
        with open(filename) as infile:
            for line in infile:
                ex = Example(json.loads(line), self.vocab, use_bias=False)
                dataset.append(ex)
        
        features = np.stack(list(ex.x for ex in dataset))
        label = np.stack(list(np.array([ex.y]) for ex in dataset))

        self.feature = torch.from_numpy(features.astype(np.float32))
        self.label = torch.from_numpy(label.astype(np.float32))
        self.num_samples = len(self.label)
        assert self.num_samples == len(self.feature)
        None

class LogRegBuzzer(nn.Module):
    def __init__(self, num_features):
        super(LogRegBuzzer, self).__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

    def evaluate(self, data):
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
