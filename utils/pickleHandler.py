import pickle

path = "models/"


def write_idf_to_file(idf, filename):
    with open(path+filename, 'wb') as handle:
        pickle.dump(idf, handle)


def read_idf_from_file(filename):
    with open(path+filename, 'rb') as handle:
        b = pickle.loads(handle.read())
    return b
