# -*- coding: utf-8 -*-

from utils import document, term_frequency as tf
from collections import Counter


def create_model():
    documents = document.read_all_documents()
    print(len(documents))
    all_tokenized_docs = []

    i = 0
    for doc in documents:
        # doc_tokens, doc_stemmed_tokens = document.tokenize_document(doc)
        doc_tokens, doc_stemmed_tokens = document.tokenize_document_v2(doc)
        tokens = dict(Counter(doc_stemmed_tokens))

        all_tokenized_docs.append(tokens)

        if i%1000 == 0:
            print("Status: tokenizing- ", i)
        i += 1

    # print all_tokenized_docs
    print("Tokenizing Completed.")

    all_words = document.get_all_words(all_tokenized_docs)
    idf = tf.get_inverse_document_frequency(all_words, all_tokenized_docs)

    print("IDF created.")

    return dict(idf), all_tokenized_docs
