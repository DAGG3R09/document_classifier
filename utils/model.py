# -*- coding: utf-8 -*-

from utils import document, term_frequency as tf
from collections import Counter


def create_model():
    documents = document.read_all_documents()
    all_tokenized_docs = []

    for doc in documents:
        doc_tokens, doc_stemmed_tokens = document.tokenize_document(doc)
        tokens = dict(Counter(doc_stemmed_tokens))

        all_tokenized_docs.append(tokens)

    # print all_tokenized_docs

    all_words = document.get_all_words(all_tokenized_docs)
    idf = tf.get_inverse_document_frequency(all_words, all_tokenized_docs)

    return dict(idf), all_tokenized_docs
