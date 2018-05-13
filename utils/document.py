# -*- coding: utf-8 -*-

from io import open
from re import sub
from os import listdir
from collections import Counter

from nltk.stem.snowball import SnowballStemmer
from nltk import sent_tokenize, word_tokenize, ngrams

from utils import constants

stemmer = SnowballStemmer("english")


def filter_stop_words(words):
    return filter(lambda w: w not in constants.stopWords, words)


def stem_all_words(words):
    stems = [stemmer.stem(w) for w in words]
    return stems


def get_all_words(all_docs):
    all_words = set()
    for doc in all_docs:
        [all_words.add(k) for k in doc.keys()]

    return all_words


def read_all_documents():
    all_docs = []
    # i = 0
    for file in listdir("training_data"):
        # if i == 2: break

        with open("training_data/"+file, 'r', encoding='utf8') as f:
            doc = f.read().lower()
            f.close()

            if file == "englishText_0_10000":
                doc = doc.split("endofarticle.")
                all_docs += doc
            else:
                all_docs.append(doc)

    return all_docs


def tokenize_document(document):
    sentences = sent_tokenize(document)
    tokens = []
    for sentence in sentences:
        sentence = sub(r'[^\w\s]', ' ', sentence)

        tokens += [w for w in filter_stop_words(word_tokenize(sentence))]

    stemmed_tokens = stem_all_words(tokens)

    return tokens, stemmed_tokens


def get_tokenized_document(filename):
    """
    Tokenizer for test data
    :param filename: Name of file presesnt in test_data directory
    :return: returns tokenized data
    """

    with open(filename, 'r', encoding='utf8') as f:
        doc = f.read().lower()
        f.close()

    # doc, stemmed = tokenize_document(doc)
    doc, stemmed = tokenize_document_v2(doc)
    doc = Counter(stemmed)

    # print doc

    return doc


def joiner(words):
    tokens = []
    for w in words:
        x = filter_stop_words(w)

        if len(x) != 2:
            continue

        x = ' '.join(x)
        tokens.append(x)

    tokens = filter(lambda x1: x1 != '', tokens)
    return tokens


def get_bigrams(sentence):
    words = word_tokenize(sentence)
    bigrams = list(ngrams(words, 2))
    bigrams = joiner(bigrams)

    return bigrams


def tokenize_document_v2(document):
        sentences = sent_tokenize(document)
        tokens = []
        for sentence in sentences:
            sentence = sub(r'[^\w\s]', ' ', sentence)
            bigrams = get_bigrams(sentence)
            tokens += [w for w in filter_stop_words(word_tokenize(sentence))]
            tokens += bigrams

        stemmed_tokens = stem_all_words(tokens)
        return tokens, stemmed_tokens

