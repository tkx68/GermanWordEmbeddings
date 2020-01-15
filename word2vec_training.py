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

import argparse
import codecs
import logging
import multiprocessing as mp

import gensim

# configuration
parser = argparse.ArgumentParser(description='Script for training word vector models using preprocessed corpora')
parser.add_argument('corpus', type=str,
                    help='Preprocessed corpus file (one sentence plain text per line in each file)')
parser.add_argument('target', type=str, help='target file name to store model in')
parser.add_argument('-s', '--size', type=int, default=100, help='dimension of word vectors')
parser.add_argument('-w', '--window', type=int, default=5, help='size of the sliding window')
parser.add_argument('-m', '--min_count', type=int, default=5,
                    help='minimum number of occurences of a word to be considered')
parser.add_argument('-t', '--threads', type=int, default=mp.cpu_count(),
                    help='number of worker threads to train the model')
parser.add_argument('-g', '--sg', type=int, default=1, help='training algorithm: Skip-Gram (1), otherwise CBOW (0)')
parser.add_argument('-i', '--hs', type=int, default=1, help='use of hierachical sampling for training')
parser.add_argument('-n', '--negative', type=int, default=0,
                    help='use of negative sampling for training (usually between 5-20)')
parser.add_argument('--ns_exponent', type=float, default=0.75,
                    help='''Exponent used to shape the negative sampling distribution which is usuallly set to 0.75 
                            but other values esp. negative values may perform better''')
parser.add_argument('-o', '--cbow_mean', type=int, default=0,
                    help='for CBOW training algorithm: use sum (0) or mean (1) to merge context vectors')
parser.add_argument('-l', '--log_to_file', type=bool, default=False, help='write log texts to file?')
parser.add_argument('--keep', type=bool, default=False,
                    help='Keep preprocessed corpus file with ending CORPORA.corpus?')
args = parser.parse_args()

if args.log_to_file:
    log_file_name = args.target.strip() + '.log'
else:
    log_file_name = None
logging.basicConfig(
    filename=log_file_name,
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

logging.info("Using up to {} CPUs".format(args.threads))


class FileCorpus(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def __iter__(self):
        line_number = 0
        with codecs.open(self.file_name, encoding="utf-8") as f:
            for line in f:
                line_number += 1
                if line_number % 100000 == 0:
                    logging.info("Corpus line number {}: {}".format(line_number, line))
                yield line.split()


logging.info("Train word2vec model...")
# see https://radimrehurek.com/gensim/models/word2vec.html
model = gensim.models.Word2Vec(
    FileCorpus(args.corpus),
    size=args.size,
    window=args.window,
    min_count=args.min_count,
    workers=args.threads,
    sg=args.sg,
    hs=args.hs,
    negative=args.negative,
    cbow_mean=args.cbow_mean,
    ns_exponent=args.ns_exponent,
    compute_loss=True)

logging.info("Store word2vec model to {}...".format(args.target))
model.save(args.target)
model.wv.save(args.target + '.wv')
