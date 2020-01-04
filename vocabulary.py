#!/usr/bin/env python
# -*- coding: utf-8 -*-

# script to create vocabulary of given model
# 
# @author: Andreas Mueller
# @see: Bachelor Thesis 'Analyse von Wort-Vektoren deutscher Textkorpora'
#
# Contributors:
#  Michael Egger <michael.egger@tsn.at>
#
# @example: python vocabulary.py test.model test.model.vocab

import gensim
import argparse
import codecs
import ProgressBar

# configuration
parser = argparse.ArgumentParser(description='Script for computing vocabulary of given corpus')
parser.add_argument('model', type=str, help='source file with trained model')
parser.add_argument('target', type=str, help='target file name to store vocabulary in')
args = parser.parse_args()

# load model
print u'Load word2vec model from file...'
model = gensim.models.KeyedVectors.load_word2vec_format(args.model, binary=True)

# build vocab
print u'Extract vocabulary from model...'
items = model.vocab.items()
n = len(items)
vocab = []
for i, (word, obj) in enumerate(items):
    ProgressBar.print_progress_bar(i, n)
    vocab.append([word, obj.count])

# save vocab
print u'Write vocablulary...'
n = len(vocab)
with codecs.open(args.target, 'w', encoding='utf-8') as f:
    for i, (word, count) in enumerate(sorted(vocab, key=lambda x: x[1], reverse=True)):
        ProgressBar.print_progress_bar(i, n)
        f.write(u'{} {}\n'.format(count, word))
