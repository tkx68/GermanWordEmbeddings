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
import os
import pickle
from itertools import islice
from typing import Callable, Iterable, List

import gensim
import stanfordnlp
from nltk.corpus import stopwords
from nltk.stem.cistem import Cistem

# configuration
parser = argparse.ArgumentParser(description='Script for training word vector models using public corpora')
parser.add_argument('corpora', type=str,
                    help='source folder with preprocessed corpora (one sentence plain text per line in each file)')
parser.add_argument('--ngrams', type=int, default=2, help='use n-grams for n = 1, 2 or 3?')
parser.add_argument('-x', '--eliminate_stopwords', type=bool, default=True, help='eliminate stop words?')
parser.add_argument('--lemmatize', type=bool, default=False, help='use only lemmata of words?')
parser.add_argument('--stem', type=bool, default=False, help='use only stems of words via nltk''s CISTEM stemmer?')
parser.add_argument('-t', '--threads', type=int, default=mp.cpu_count(),
                    help='number of worker threads to train the model')
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

logging.info("Using up to {} CPUs".format(args.threads))
if args.lemmatize:
    logging.info("Perform lemmatisation using nltk")
    nlp_de = stanfordnlp.Pipeline(lang="de", processors="tokenize,lemma", use_gpu=True)
if args.stem:
    logging.info("Perform stemming using nltk''s CISTEM stemmer")
    stemmer = Cistem()

stop_words = stopwords.words('german')


# get corpus sentences
class CorpusSentences(object):
    def __init__(self, directory_name: str, chunk_size: int = 10000):
        self.directory_name = directory_name
        self.n = chunk_size

    def __iter__(self):
        for file_name in os.listdir(self.directory_name):
            logging.info("Use corpus file %s " % file_name)
            with codecs.open(os.path.join(self.directory_name, file_name), encoding="utf-8") as f:
                while True:
                    lines: List[str] = list(islice(f, self.n))
                    if not lines:
                        break
                    if args.stem:
                        for line in lines:
                            sentence = line.split()
                            stemmed_sentence = [stemmer.stem(w) for w in sentence]
                            logging.debug("Stemmed sentence: %s" % ' '.join(stemmed_sentence))
                            yield stemmed_sentence
                    elif args.lemmatize:
                        # one CR is already there and we add another one according to stanfordnlp's hint:
                        doc = u'\n'.join(lines)
                        logging.debug("Lemmatize the following block: ", doc)
                        n = nlp_de(doc)
                        for sentence in n.sentences:
                            lemmatized_sentence = [t.words[0].lemma for t in sentence.tokens if t.words[0].lemma is not None and not t.words[0].lemma in stop_words]
                            logging.debug("Yield lemmatized sentence: %s" % ' '.join(lemmatized_sentence))
                            yield lemmatized_sentence
                    else:
                        for line in lines:
                            yield line.split()


class Filter(object):
    def __init__(self, corpus: Iterable[List[str]], predicate: Callable[[str], bool]):
        self.corpus = corpus
        self.predicate = predicate

    def __iter__(self):
        for sentence in self.corpus:
            yield [word for word in sentence if self.predicate(word)]


def is_not_stop_word(word: str) -> bool:
    return not (word in stop_words)


bigram_transformer = None
trigram_transformer = None
if args.ngrams > 1:
    logging.info("Build bigram transformer...")
    bigram_transformer = gensim.models.Phrases(CorpusSentences(args.corpora), common_terms=stop_words)
    with open(args.target + u'.bigram', mode="wb") as f:
        pickle.dump(bigram_transformer, f)
    sentences: Iterable[List[str]]
    sentences = bigram_transformer[CorpusSentences(args.corpora)]
    if args.ngrams > 2:
        sentences2 = list(sentences)
        bigram_transformer = None
        logging.info("Build trigram transformer from sentences with unigrams and bigrams...")
        trigram_transformer = gensim.models.Phrases(sentences, common_terms=stop_words)
        with open(args.target + u'.trigram', mode="wb") as f:
            pickle.dump(trigram_transformer, f)
        sentences = trigram_transformer[sentences2]
else:
    sentences = CorpusSentences(args.corpora)

if args.eliminate_stopwords:
    logging.info("Introduce stop words filter...")
    sentences = Filter(sentences, is_not_stop_word)

with codecs.open(args.target + '.corpus', 'w', encoding="utf-8") as tmp:
    i: int
    s: List[str]
    for i, s in enumerate(sentences):
        if i and i % 100000 == 0:
            logging.info('Saved {} k sentences to temporary file'.format(round(i / 1000)))
            tmp.flush()
        tmp.write(' '.join(s) + u'\n')
