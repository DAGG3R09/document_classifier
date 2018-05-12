# -*- coding: utf-8 -*-

from utils import document, term_frequency as tf
from collections import Counter

if __name__ == "__main__":
    documents = document.read_all_documents()
    all_tokenized_docs = []

    for doc in documents:
        doc_tokens, doc_stemmed_tokens = document.tokenize_document(doc)
        tokens = dict(Counter(doc_stemmed_tokens))

        all_tokenized_docs.append((tokens))

    print(all_tokenized_docs, "\n\n")
    all_words = document.get_all_words(all_tokenized_docs)
    idf = tf.get_inverse_document_frequency(all_words, all_tokenized_docs)

    words = tf.tfidf_one(idf, all_tokenized_docs[0])

    for word, value in words:
        print(word, value)