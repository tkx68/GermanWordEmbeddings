#!/usr/bin/env python
# -*- coding: utf-8 -*-

# script to train word embeddings with word2vec
# 
# @author: Andreas Mueller
# @see: Bachelor Thesis 'Analyse von Wort-Vektoren deutscher Textkorpora'
#
# Contributors:
#  Michael Egger <michael.egger@tsn.at>
#
# @example: python word2vec_training.py corpus_dir/ test.model -s 300 -w 10

import gensim
from gensim.models.doc2vec import TaggedDocument
import logging
import os
import argparse
import multiprocessing as mp
import codecs

# configuration
parser = argparse.ArgumentParser(description='Script for training doc vector models using preprocessed corpora')
parser.add_argument('corpus', type=str,
                    help='Preprocessed corpus file (one sentence plain text per line in each file)')
parser.add_argument('target', type=str, help='target file name to store model in')
parser.add_argument('-s', '--size', type=int, default=100, help='dimension of word vectors')
parser.add_argument('-w', '--window', type=int, default=5, help='size of the sliding window')
parser.add_argument('-m', '--mincount', type=int, default=5,
                    help='minimum number of occurences of a word to be considered')
parser.add_argument('-t', '--threads', type=int, default=mp.cpu_count(),
                    help='number of worker threads to train the model')
parser.add_argument('-a', '--algorithm', type=int, default=1,
                    help='training algorithm: If 1, "distributed memory" (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.')
parser.add_argument('--epochs', type=int, default=5, help='number of iterations (epochs) of training through the corpus')
parser.add_argument('--alpha', type=int, default=0.025, help='starting value for the learning rate')
parser.add_argument('--min_alpha', type=int, default=0.0001, help='minimum value for the learning rate')
parser.add_argument('--hs', type=int, default=1, help='use of hierachical sampling for training')
parser.add_argument('--negative', type=int, default=0,
                    help='use of negative sampling for training (usually between 5-20)')
args = parser.parse_args()
logging.basicConfig(
    filename=args.target.strip() + '.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO
)

logging.info("Using {} CPUs".format(args.threads))


# get corpus sentences
class CorpusDocuments(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def __iter__(self):
        line_number = 0
        with codecs.open(self.file_name, encoding="utf-8") as f:
            for line in f:
                line_number += 1
                if line_number % 100000 == 0:
                    logging.info("Corpus line number {}: {}".format(line_number, line))
                yield TaggedDocument(line.split(), [line_number])


logging.info("Train word2vec model...")
documents = CorpusDocuments(args.corpus)
model = gensim.models.Doc2Vec(documents,
                              vector_size=args.size,
                              window=args.window,
                              min_count=args.mincount,
                              workers=args.threads,
                              dm=args.algorithm,
                              hs=args.hs,
                              negative=args.negative,
                              alpha=args.alpha,
                              min_alpha=args.min_alpha)

# store model
logging.info("Store doc2vec model to {}...".format(args.target))
model.save(args.target)
