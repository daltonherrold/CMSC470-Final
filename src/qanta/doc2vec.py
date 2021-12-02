from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
from os import path

import click
from flask import Flask, jsonify, request

from qanta import util
from qanta.dataset import QuizBowlDataset

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize


MODEL_PATH = 'doc2vec.pickle'
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.3


def guess_and_buzz(model, question_text) -> Tuple[str, bool]:
    guesses = model.guess([question_text], BUZZ_NUM_GUESSES)
    scores = [guess[1] for guess in guesses]
    buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
    return guesses[0][0], buzz


class Doc2VecGuesser:
    def __init__(self):
        self.model = None
        self.i_to_ans = None

    def train(self, training_data) -> None:
        x_array = training_data[0]
        y_array = training_data[1]
        x_array = [' '.join(x) for x in x_array]
        # questions = training_data[0]
        # answers = training_data[1]
        # answer_docs = defaultdict(str)
        # for q, ans in zip(questions, answers):
        #     text = ' '.join(q)
        #     answer_docs[ans] += ' ' + text

        # x_array = []
        # y_array = []
        # for ans, doc in answer_docs.items():
        #     x_array.append(doc)
        #     y_array.append(ans)

        self.i_to_ans = {i: ans for i, ans in enumerate(y_array)}
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(x_array)]
        
        # max_epochs = 100
        # vec_size = 20
        # alpha = 0.025

        self.model = Doc2Vec(tagged_data, workers=4)
        
        # model.build_vocab(tagged_data)

        # for epoch in range(max_epochs):
        #     print('iteration {0}'.format(epoch))
        #     model.train(tagged_data,
        #                 total_examples=model.corpus_count,
        #                 epochs=model.iter)
        #     # decrease the learning rate
        #     model.alpha -= 0.0002
        #     # fix the learning rate, no decay
        #     model.min_alpha = model.alpha

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        test = word_tokenize(questions[0].lower())
        ivec = self.model.infer_vector(test)

        g = self.model.docvecs.most_similar(positive=[ivec], topn=max_n_guesses)
    
        guesses = [(self.i_to_ans[int(i)], n) for i, n in g]

        return guesses

    def save(self):
        with open("model.pickle", 'wb') as f:
            self.model.save("model.pickle")
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'i_to_ans': self.i_to_ans,
            }, f)

    @classmethod
    def load(cls):
        guesser = Doc2VecGuesser()
        guesser.model = Doc2Vec.load("model.pickle")
        with open(MODEL_PATH, 'rb') as f:
            params = pickle.load(f)
            guesser.i_to_ans = params['i_to_ans']
            return guesser



def create_app(enable_batch=True):
    doc2vec_guesser = Doc2VecGuesser.load()
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        question = request.json['text']
        guess, buzz = guess_and_buzz(doc2vec_guesser, question)
        return jsonify({'guess': guess, 'buzz': True if buzz else False})

    @app.route('/api/1.0/quizbowl/status', methods=['GET'])
    def status():
        return jsonify({
            'batch': False,
            'batch_size': 200,
            'ready': True,
            'include_wiki_paragraphs': False
        })


    return app


@click.group()
def cli():
    pass


@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
@click.option('--disable-batch', default=False, is_flag=True)
def web(host, port, disable_batch):
    """
    Start web server wrapping tfidf model
    """
    app = create_app(enable_batch=not disable_batch)
    app.run(host=host, port=port, debug=False)


@cli.command()
def train():
    """
    Train the tfidf model, requires downloaded data and saves to models/
    """
    dataset = QuizBowlDataset(guesser_train=True)
    doc2vec_guesser = Doc2VecGuesser()
    doc2vec_guesser.train(dataset.training_data())
    doc2vec_guesser.save()


@cli.command()
@click.option('--local-qanta-prefix', default='data/')
@click.option('--retrieve-paragraphs', default=False, is_flag=True)
def download(local_qanta_prefix, retrieve_paragraphs):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    util.download(local_qanta_prefix, retrieve_paragraphs)


if __name__ == '__main__':
    cli()
