import numpy as np


def group(X_data, Y_data, as_array: bool = True):
    grouped = dict()
    for x, y in zip(X_data, Y_data):
        if y not in grouped:
            grouped[y] = []
        grouped[y].append(x)
    return grouped if not as_array else dict((k, np.array(v)) for k, v in grouped.items())


def xor(a: bool, b: bool):
    return (a and not b) or (not a and b)
