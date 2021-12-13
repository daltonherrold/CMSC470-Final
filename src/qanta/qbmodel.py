from typing import List, Tuple
from gensim.models import doc2vec
import nltk
import sklearn
import transformers
import numpy as np
import pandas as pd
import torch
from doc2vec import Doc2VecGuesser
from doc2vec import guess_and_buzz as doit
from buzzer import LogRegBuzzer


class QuizBowlModel:

    def __init__(self):
        """
        Load your model(s) and whatever else you need in this function.
        Do NOT load your model or resources in the guess_and_buzz() function, 
        as it will increase latency severely. 
        """
        self.doc2vec_guesser = Doc2VecGuesser.load()

        with open('vocab', 'r') as infile:
            self.vocab = [x.strip() for x in infile]

        self.buzzer = LogRegBuzzer(len(self.vocab))
        self.buzzer.load_state_dict(torch.load('trained_model.th'))

    def guess_and_buzz(self, question_text: List[str]) -> List[Tuple[str, bool]]:
        """
        This function accepts a list of question strings, and returns a list of tuples containing
        strings representing the guess and corresponding booleans representing 
        whether or not to buzz. 
        So, guess_and_buzz(["This is a question"]) should return [("answer", False)]
        If you are using a deep learning model, try to use batched prediction instead of 
        iterating using a for loop.
        """
        return [doit(self.doc2vec_guesser, x, self.vocab, self.buzzer) for x in question_text]