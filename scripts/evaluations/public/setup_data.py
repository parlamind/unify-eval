import os
from unify_eval.utils.corpus import Corpus


"""
Use this script to fetch any of the public corpora from the corresponding JSON file
and have it in form of a Corpus object (ready to be used for training)
"""


def get_data(data, test_data):
    """
    fetch the desired corpus and have it in form of a Corpus object
    :param data: one of {"snips", "AskUbuntuCorpus", "ChatbotCorpus", "WebApplicationsCorpus"}
    :param test_data: if true, it fetches the test set of that corpus, otherwise the training set
    :return: a Corpus object with the test / training set of the specified corpus
    """
    if data == "snips":
        return get_snips_corpus(test_data)


def get_snips_corpus(test_data):
    name = "test" if test_data else "train"
    corpus = Corpus.fromJSON(path=os.path.join("data", "public", "snips", name + ".json"))
    return corpus
