from collections import Counter, OrderedDict
from math import log
from operator import itemgetter


total_documents = 10000
new_idf = log(total_documents)

def get_term_frequency_matrix(tokenized_docs):
    tf = {}

    for i, doc in enumerate(tokenized_docs):
        tf[i] = dict(Counter(doc))

    return tf


def get_inverse_document_frequency(all_words, all_docs):
    """
        idf of a word = log (Occurrence of word in all documents / number of documents word occurred in.)
        :param all_words: All the words mentioned in all the documents
        :param all_docs: All the tokenized documents
        :return: {dictionary} of the Inverse Document frequencies of every word.

    """

    idf = OrderedDict()

    for word in all_words:
        count = 0
        number_of_documents = 0
        for doc in all_docs:
            c = doc.get(word, 0)
            if c:
                number_of_documents += 1
        idf[word] = log(total_documents / number_of_documents)

    return idf


def tfidf_one(idf, doc):
    """

    :param idf: The Inverse Document Frequency Matrix
    :param doc: The tokenized document to be analyzed
    :return: The important words of the document
    """

    tf_idf = {}
    for word, count in doc.items():
        tf_idf[word] = count * idf.get(word, new_idf)

    return sorted(tf_idf.items(), key=itemgetter(1), reverse=True)[:10]