# -*- coding: utf-8 -*-

from io import open
from re import sub
from os import listdir
from collections import Counter

from nltk.stem.snowball import SnowballStemmer
from nltk import sent_tokenize, word_tokenize

from utils import constants

stemmer = SnowballStemmer("english")


def filter_stop_words(words):
    return filter(lambda w: w not in constants.stopWords, words)


def stem_all_words(words):
    stems = [stemmer.stem(w) for w in words]
    return stems


def tokenize_document(document):
    sentences = sent_tokenize(document)
    tokens = []
    for sentence in sentences:
        sentence = sub(r'[^\w\s]', ' ', sentence)

        tokens += [w for w in filter_stop_words(word_tokenize(sentence))]

    tokens = filter(lambda t: t != u'the', tokens)
    stemmed_tokens = stem_all_words(tokens)


    return tokens, stemmed_tokens


def read_all_documents():
    all_docs = []

    for file in listdir("training_data"):

        with open("training_data/"+file, 'r', encoding='utf8') as f:
            doc = f.read().lower()
            f.close()
            all_docs.append(doc)

    return all_docs


def get_all_words(all_docs):
    all_words = set()
    for doc in all_docs:
        [all_words.add(k) for k in doc.keys()]

    return all_words


def get_tokenized_document(filename):
    with open(filename, 'r', encoding='utf8') as f:
        doc = f.read().lower()
        f.close()

    doc, stemmed = tokenize_document(doc)
    doc = Counter(stemmed)

    return doc
