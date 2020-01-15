#!/usr/bin/env python
# -*- coding: utf-8 -*-

# script to preprocess corpora for training
#
# @author: Andreas Mueller
# @see: Bachelor Thesis 'Analyse von Wort-Vektoren deutscher Textkorpora'
#
# Contributors:
#  Michael Egger <michael.egger@tsn.at>
#
# @example: python cleansing.py test.raw test.corpus -psub

import codecs
import gensim
import nltk.data
import argparse
import os
import re
import logging
import sys
import multiprocessing as mp

# configuration
parser = argparse.ArgumentParser(description='Script for cleaning public corpora')
parser.add_argument('raw', type=str, help='source file with raw data for corpus creation')
parser.add_argument('target', type=str, help='target file name to store corpus in')
parser.add_argument('-w', '--min_words_per_sentence', type=int, default=2,
                    help='minimum number of words per sentence')  # mp.cpu_count()
parser.add_argument('-p', '--punctuation', action='store_true', help='remove punctuation tokens')
parser.add_argument(
    '-u', '--umlauts', action='store_true', help='replace german umlauts with their respective digraphs'
)
parser.add_argument('-t', '--threads', type=int, default=8, help='thread count')  # mp.cpu_count()
parser.add_argument('--batch_size', type=int, default=32, help='batch size for multiprocessing')
parser.add_argument('-l', '--log_to_file', type=bool, default=False, help='write log texts to file?')
args = parser.parse_args()

if args.log_to_file:
    log_file_name = args.target.strip() + '.log'
else:
    log_file_name = None
logging.basicConfig(stream=log_file_name,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

logging.info("Pre-processing file {}".format(args.raw))

sentence_detector = nltk.data.load('tokenizers/punkt/german.pickle')
punctuation_tokens = [u'.', u'..', u'...', u',', u';', u':', u'(', u')', u'"', u'\'', u'[', u']',
                      u'{', u'}', u'?', u'!', u'-', u'–', u'+', u'*', u'--', u'\'\'', u'``']
punctuation = u'?.!/;:()&+'


def replace_umlauts(text):
    """
    Replaces german umlauts and sharp s in given text.

    :param text: text as str
    :return: manipulated text as str
    """
    res = text
    res = res.replace(u'ä', u'ae')
    res = res.replace(u'ö', u'oe')
    res = res.replace(u'ü', u'ue')
    res = res.replace(u'Ä', u'Ae')
    res = res.replace(u'Ö', u'Oe')
    res = res.replace(u'Ü', u'Ue')
    res = res.replace(u'ß', u'ss')
    return res


def process_line(line):
    """
    Pre processes the given line.

    :param line: line as str
    :return: preprocessed sentence
    """
    if line == u"":
        return u""
    # detect sentences
    sentences = sentence_detector.tokenize(line)
    # process each sentence
    for sentence in sentences:
        # replace umlauts
        if args.umlauts:
            sentence = replace_umlauts(sentence)
        # get word tokens
        words = nltk.word_tokenize(sentence)
        # filter punctuation and stopwords
        if args.punctuation:
            words = [x for x in words if x not in punctuation_tokens]
            words = [re.sub(u'[' + punctuation + u']', u'', x) for x in words]
        # write one sentence per line in output file, if sentence has more than min_words_per_sentence words
        if len(words) >= args.min_words_per_sentence:
            words = u' '.join(words) + u'\n'
            return words


if not os.path.exists(os.path.dirname(args.target)):
    os.makedirs(os.path.dirname(args.target))

with codecs.open(args.target, 'w', encoding="utf-8") as outfile:
    with codecs.open(args.raw, 'r', encoding="utf-8") as infile:
        # start pre processing with multiple threads
        pool = mp.Pool(args.threads)
        values = pool.imap(process_line, infile, chunksize=args.batch_size)
        for i, s in enumerate(values):
            if i and i % 100000 == 0:
                logging.info('processed {} k sentences'.format(round(i / 1000)))
                outfile.flush()
            if s:
                outfile.write(s)
        logging.info('preprocessing of {} sentences finished!'.format(i))
