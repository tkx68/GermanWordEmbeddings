#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import pickle
import re
from datetime import date
from math import sqrt

import gensim
import pandas as pd
import psycopg2 as pg
from nltk.corpus import stopwords

# configuration
parser = argparse.ArgumentParser(description='Smart search of key words in ConZone profiles')
parser.add_argument('model', type=str,
                    help='word2vec model file with word vectors only')
parser.add_argument('--target', type=str,
                    help='target file name without extension to store search results in; default is to use the query as target file name')
parser.add_argument('-q', '--query', type=str, help='search query')
parser.add_argument('-p', '--partner', type=str, default='Xing', help='ConZone partner name to search in')
parser.add_argument('-s', '--sep', type=str, default=';', help='separator for CSV file')
parser.add_argument('-f', '--format', type=str, default='xlsx', choices=['xlsx', 'csv'], help='output format')
parser.add_argument('-l', '--log_to_file', type=bool, default=False, help='write log texts to file?')
args = parser.parse_args()

if args.query is None or args.model is None:
    print(parser.format_usage())
    quit()

if args.target is None:
    args.target = u'Query ' + args.query

if args.log_to_file:
    log_file_name = args.target.strip() + '.log'
else:
    log_file_name = None
logging.basicConfig(
    filename=log_file_name,
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

logging.info("Compute word movers distance of ConZone employments to '%s'" % (args.query,))

mwv = gensim.models.KeyedVectors.load(args.model, mmap='r')
file_name_wo_ext = os.path.splitext(args.model)[0]
if os.path.isfile(file_name_wo_ext + u'.bigram'):
    logging.info("Loading bigram transformer...")
    with open(file_name_wo_ext + u'.bigram', mode="rb") as f:
        bigram_transformer = pickle.load(f)
    if os.path.isfile(file_name_wo_ext + u'.trigram'):
        ngram = 3
        logging.info("Loading trigram transformer...")
        with open(file_name_wo_ext + u'.trigram', mode="rb") as f:
            trigram_transformer = pickle.load(f)
    else:
        ngram = 2
else:
    ngram = 1

logging.info("ngram level is %s" % (ngram,))


def ngram_transform(sentence):
    if ngram == 3:
        return trigram_transformer[bigram_transformer[sentence]]
    elif ngram == 2:
        return bigram_transformer[sentence]
    else:
        return sentence


stop_words = stopwords.words('german')
punctuation_tokens = [u'.', u'..', u'...', u',', u';', u':', u'(', u')', u'"', u'\'', u'[', u']', u'|',
                      u'{', u'}', u'?', u'!', u'-', u'–', u'+', u'*', u'--', u'\'\'', u'``']
punctuation = u'?.!/;:()&+|'


def cleanup_punctuation(text):
    sentence = text.replace("\n", " ").split()
    sentence = [w for w in sentence if w not in punctuation_tokens]
    sentence = [re.sub(u'[' + punctuation + u']', u'', w) for w in sentence]
    return sentence


def remove_stop_words(sentence):
    return [w for w in sentence if w not in stop_words]


query = remove_stop_words(ngram_transform(args.query.split()))

try:
    logger = logging.getLogger('ConZone')
    connect_str = "host=127.0.0.1 port=5433 dbname=conzone user=conzone password=conzone1234"
    logger.debug("Open connection to conzone DB with %s", connect_str)
    con = pg.connect(connect_str)
    cur = con.cursor()
    d = pd.DataFrame()
    today = date.today()
    cur.execute("SELECT id from partner where name=%s;", (args.partner,))
    partner_id = cur.fetchone()[0]
    logger.info("We collect data from the partner named %s with id = %s", args.partner, partner_id)
    cur.execute("select id from supplier where partner_id = %s", (partner_id,))
    suppliers = cur.fetchall()
    for supplier in suppliers:
        supplier_id = supplier[0]
        logger.info("Supplier id = %s", supplier_id)
        cur.execute(
            "select id, first_name, second_name, last_name, date_of_birth from consultant where supplier_id = %s",
            (supplier_id,))
        consultants = cur.fetchall()
        for c in consultants:
            logger.debug("Consultant: %s", c)
            cur.execute(
                """select employer, employment_start, employment_end, city, tasks, c.name as country
                from employment e left outer join country c on e.country=c.id
                where consultant_id = %s""",
                (c[0],))
            date_of_birth = c[4]
            employments = cur.fetchall()
            logger.debug("Found %s employments", len(employments))
            for empl in employments:
                employer = remove_stop_words(ngram_transform(cleanup_punctuation(empl[0])))
                employment_start = empl[1]
                employment_end = empl[2] if empl[2] is not None else today
                employment_end = min(employment_end, today)
                duration_factor = sqrt(
                    (employment_end - employment_start).days / 365.0) if employment_start.year != 1 else 1
                duration_factor = min(3.0, max(0.0, duration_factor))
                employment_age = (today - employment_end).days / 365.0
                aging_factor = 3 / (3 + sqrt(employment_age))
                wmd_employer = mwv.wmdistance(employer, query)
                tasks = empl[4] if empl[4] != "<Unknown>" else ""
                tasks = remove_stop_words(ngram_transform(cleanup_punctuation(tasks)))
                if len(tasks) > 0:
                    logger.debug("Found meaningful tasks: %s" % tasks)
                    wmd_tasks = mwv.wmdistance(tasks, query)
                    total_score = aging_factor * (1 / wmd_tasks * 1 / wmd_employer) ** duration_factor
                    x = {
                        'query': str(query),
                        'consultant_id': c[0],
                        'first_name': c[1],
                        'second_names': c[2],
                        'last_name': c[3],
                        'date_of_birth': date_of_birth,
                        'employer': empl[0],
                        'cleaned_employer': u' '.join(employer),
                        'wmd_employer': wmd_employer,
                        'employment_start': employment_start.strftime(
                            "%d.%m.%Y") if employment_start is not None else "",
                        'employment_end': employment_end.strftime("%d.%m.%Y") if employment_end is not None else "",
                        'duration_factor': duration_factor,
                        'aging_factor': aging_factor,
                        'employment_city': empl[3] if empl[3] is not None else "",
                        'employment_country': empl[5] if empl[5] is not None else "",
                        'tasks': empl[4],
                        'cleaned_tasks': u' '.join(tasks),
                        'wmd_tasks': wmd_tasks,
                        'total_score': total_score
                    }
                    d = d.append(pd.DataFrame(x, index=[0]), ignore_index=True)
    logger.info("Writing to output file %s..." % (args.target + u'.' + args.format))
    if args.format == 'xlsx':
        d.to_excel(args.target + u'.xlsx')
    elif args.format == 'csv':
        d.to_csv(args.target + u'.csv', sep=args.sep)
    else:
        logger.warning("No output format given.")
finally:
    logger.info("Now closing connection")
    con.close()
    logger.info("END")
