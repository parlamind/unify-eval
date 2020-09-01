from typing import List

from unify_eval.model.types import Label
from unify_eval.utils.corpus import Corpus


"""
Utility methods for data pre-processing
"""


def rewrite_entry(clause: str, label: Label, regular: bool) -> dict:
    """
    given a clause and a label, produce a dict object where the keys match exactly
    what is expected in the JSON format used in Unifyeval
    :param clause:
    :param label:
    :param regular: if false, then the format for isolated evaluation is used
    :return:
    """
    if regular:
        return dict(clause=clause, label=label)
    return dict(clauses=[clause],
                goldIndexIds=[label] if label not in {-1, "-1"} else [],
                subject="")


def rewrite_corpus(corpus: Corpus, regular=False) -> List[dict]:
    return [rewrite_entry(clause=clause, label=label, regular=regular) for clause, label in zip(corpus.X, corpus.Y)]