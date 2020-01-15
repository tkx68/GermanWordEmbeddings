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
import gc
import logging
import multiprocessing as mp
import os
import pickle
from itertools import islice
from typing import Callable, Iterable, List

import gensim
import stanfordnlp
from nltk.corpus import stopwords

# configuration
parser = argparse.ArgumentParser(description='Script for lemmatizing corpora')
parser.add_argument('corpora', type=str,
                    help='source folder with preprocessed corpora (one sentence plain text per line in each file)')
parser.add_argument('target', type=str, help='target folder name to store lemmatized documents in')
parser.add_argument('-l', '--log_to_file', type=bool, default=False, help='write log texts to file?')
args = parser.parse_args()

if args.log_to_file:
    log_file_name = args.target.strip() + '.log'
else:
    log_file_name = None
logging.basicConfig(
    filename=log_file_name,
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

logging.info("Perform lemmatisation using stanfordnlp")
nlp_de = stanfordnlp.Pipeline(lang="de", processors="tokenize,lemma", use_gpu=True)

i = 1
for file_name in os.listdir(args.corpora):
    logging.info("Lemmatize corpus file %s " % file_name)
    with codecs.open(os.path.join(args.corpora, file_name), encoding="utf-8") as in_file:
        with codecs.open(os.path.join(args.target, file_name), mode="wb", encoding="utf-8") as out_file:
            while True:
                lines: List[str] = list(islice(in_file, 100000))
                if not lines:
                    break
                doc = u'\n'.join(lines) # one CR is already there and we add another one according to stanfordnlp's hint.
                logging.debug("Lemmatize the following block: ", doc)
                n = nlp_de(doc)
                for sentence in n.sentences:
                    lemmatized_sentence = [t.words[0].lemma for t in sentence.tokens if t.words[0].lemma is not None]
                    line = ' '.join(lemmatized_sentence)
                    logging.info("Write lemmatized sentence #%s: %s" % (i, line))
                    i += 1
                    out_file.write(line + u'\n')
