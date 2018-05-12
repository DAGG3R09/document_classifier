
from sys import argv

from utils.model import create_model
from utils.pickleHandler import read_idf_from_file, write_idf_to_file

from utils.document import get_tokenized_document

from utils import term_frequency as tf
if __name__ == "__main__":

    if len(argv) == 2:
        idf = read_idf_from_file(argv[1])
    else:
        name = raw_input("Enter Name for Model: ")
        idf, all_docs = create_model()
        write_idf_to_file(idf, name)

    doc_name = raw_input("Enter File name in test_data: ")
    doc = get_tokenized_document("test_data/"+doc_name)

    words = tf.tfidf_one(idf, doc)

    for word, value in words:
        print(word, value)