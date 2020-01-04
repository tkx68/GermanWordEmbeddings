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
# @example: python training.py corpus_dir/ test.model -s 300 -w 10

import gensim
import logging
import os
import argparse
import multiprocessing as mp

# configuration
parser = argparse.ArgumentParser(description='Script for training word vector models using public corpora')
parser.add_argument('corpora', type=str,
                    help='source folder with preprocessed corpora (one sentence plain text per line in each file)')
parser.add_argument('target', type=str, help='target file name to store model in')
parser.add_argument('-s', '--size', type=int, default=100, help='dimension of word vectors')
parser.add_argument('-w', '--window', type=int, default=5, help='size of the sliding window')
parser.add_argument('-m', '--mincount', type=int, default=5,
                    help='minimum number of occurences of a word to be considered')
parser.add_argument('-t', '--threads', type=int, default=mp.cpu_count(),
                    help='number of worker threads to train the model')
parser.add_argument('-g', '--sg', type=int, default=1, help='training algorithm: Skip-Gram (1), otherwise CBOW (0)')
parser.add_argument('-i', '--hs', type=int, default=1, help='use of hierachical sampling for training')
parser.add_argument('-n', '--negative', type=int, default=0,
                    help='use of negative sampling for training (usually between 5-20)')
parser.add_argument('-o', '--cbowmean', type=int, default=0,
                    help='for CBOW training algorithm: use sum (0) or mean (1) to merge context vectors')
args = parser.parse_args()
logging.basicConfig(
    filename=args.target.strip() + '.result', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO
)

logging.info("Using {} CPUs".format(args.threads))


# get corpus sentences
class CorpusSentences(object):
    def __init__(self, directory_name):
        self.directory_name = directory_name

    def __iter__(self):
        for file_name in os.listdir(self.directory_name):
            with open(os.path.join(self.directory_name, file_name)) as fp:
                for line in fp:
                    yield line.split()


sentences = CorpusSentences(args.corpora)

logging.info("Train word2vec model...")
# train the model
# see https://radimrehurek.com/gensim/models/word2vec.html
model = gensim.models.Word2Vec(
    sentences,
    size=args.size,
    window=args.window,
    min_count=args.mincount,
    workers=args.threads,
    sg=args.sg,
    hs=args.hs,
    negative=args.negative,
    cbow_mean=args.cbowmean
)

# store model
logging.info("Store word2vec model to {}...".format(args.target))
model.wv.save_word2vec_format(args.target, binary=True)
